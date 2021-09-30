use crate::bit_array::*;
use crate::instructions::*;
use crate::program;

use std::collections::BTreeSet;
use std::cell::Cell;

#[derive(Debug)]
pub enum CompileError {
    // TODO: Proper desc/fields
    // TODO: Test(s)
    TooManyRegisters,
}

#[derive(Debug)]
struct Program {
    statements: Vec<Statement>,
    variable_registers: Vec<Option<Register>>,
}

impl Program {
    fn new(num_variables: u32) -> Program {
        Program {
            statements: Vec::new(),
            variable_registers: vec![None; num_variables as usize],
        }
    }

    fn reg(&self, variable_index: VariableIndex) -> Option<Register> {
        self.variable_registers[variable_index.0 as usize]
    }
}

#[derive(Debug)]
struct Ddg {
    nodes: Vec<Node>,
    variable_def_nodes: Vec<Option<NodeIndex>>,
    output_stream_nodes: Vec<NodeIndex>,
}

impl Ddg {
    fn new(num_variables: u32) -> Ddg {
        Ddg {
            nodes: Vec::new(),
            variable_def_nodes: vec![None; num_variables as usize],
            output_stream_nodes: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct Node {
    statement: Statement,
    predecessors: Vec<NodeIndex>,
    is_scheduled: Cell<bool>,
}

impl Node {
    fn new(statement: Statement, predecessors: Vec<NodeIndex>) -> Node {
        Node {
            statement,
            predecessors,
            is_scheduled: Cell::new(false),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NodeIndex(u32);

#[derive(Debug, Clone, Copy)]
enum Statement {
    Add(VariableIndex, VariableIndex, VariableIndex),
    Load(VariableIndex, InputStream, u8),
    Multiply(VariableIndex, VariableIndex, VariableIndex, u8),
    Store(OutputStream, VariableIndex, u8),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct VariableIndex(u32);

impl From<program::VariableIndex> for VariableIndex {
    fn from(i: program::VariableIndex) -> Self {
        Self(i.0)
    }
}

#[derive(Debug)]
pub struct CompiledProgram {
    pub instructions: Box<[EncodedInstruction]>,
    pub input_stream_thread_strides: Box<[u32]>,
    pub output_stream_thread_stride: u32,
}

pub fn compile(p: &program::Program) -> Result<CompiledProgram, CompileError> {
    println!("Program statements:");
    for s in &p.statements {
        println!("  {:?}", s);
    }

    let (input_stream_thread_strides, output_stream_thread_stride) = determine_thread_strides(p);

    let ddg = construct_ddg(p);
    println!("Directed dependency graph: {:#?}", ddg);

    let mut program = Program::new(p.num_variables);

    schedule(&ddg, &mut program);
    println!("Schedule:");
    for statement in &program.statements {
        println!("  {:?}", statement);
    }

    let (interference_edges, variable_degrees) = analyze_liveness(&program);
    println!("Interference edges: {:#?}", interference_edges);
    println!("Variable degrees: {:#?}", variable_degrees.iter().enumerate().collect::<Vec<_>>());

    allocate_registers(&mut program, interference_edges, variable_degrees)?;
    println!("Variable registers: {:#?}", program.variable_registers);

    let instructions = generate_instructions(&program);
    println!("Instructions:");
    for instruction in &instructions {
        println!("  {}", instruction);
    }

    let encoded_instructions = encode_instructions(&instructions);
    println!("Encoded instructions:");
    for &encoded_instruction in &encoded_instructions {
        println!("  0x{:08x}", encoded_instruction);
    }

    Ok(CompiledProgram {
        instructions: encoded_instructions.into(),
        input_stream_thread_strides,
        output_stream_thread_stride,
    })
}

fn determine_thread_strides(p: &program::Program) -> (Box<[u32]>, u32) {
    let mut input_stream_thread_strides = Box::new([0; InputStream::VARIANT_COUNT]);

    for (&input_stream, &n) in &p.num_input_stream_loads {
        input_stream_thread_strides[input_stream as usize] = n;
    }

    let output_stream_thread_stride = p.num_output_stream_stores;

    (input_stream_thread_strides, output_stream_thread_stride)
}

fn construct_ddg(p: &program::Program) -> Ddg {
    let mut ddg = Ddg::new(p.num_variables);

    for s in &p.statements {
        visit_statement(s, &mut ddg);
    }

    ddg
}

fn visit_statement(s: &program::Statement, ddg: &mut Ddg) {
    let index = NodeIndex(ddg.nodes.len() as _);

    let (n, v) = match *s {
        program::Statement::Add(dst, ref lhs, ref rhs) => {
            let (lhs, lhs_p) = visit_scalar(lhs, ddg);
            let (rhs, rhs_p) = visit_scalar(rhs, ddg);
            (
                Node::new(
                    Statement::Add(dst.into(), lhs, rhs),
                    vec![lhs_p, rhs_p],
                ),
                Some(dst),
            )
        }
        program::Statement::Load(dst, src, offset) => {
            (
                Node::new(
                    Statement::Load(dst.into(), src, offset),
                    Vec::new(),
                ),
                Some(dst),
            )
        }
        program::Statement::Multiply(dst, ref lhs, ref rhs, shift) => {
            let (lhs, lhs_p) = visit_scalar(lhs, ddg);
            let (rhs, rhs_p) = visit_scalar(rhs, ddg);
            (
                Node::new(
                    Statement::Multiply(dst.into(), lhs, rhs, shift),
                    vec![lhs_p, rhs_p],
                ),
                Some(dst)
            )
        }
        program::Statement::Store(dst, ref src, offset) => {
            let (src, src_p) = visit_scalar(src, ddg);
            let ret = (
                Node::new(
                    Statement::Store(dst, src, offset),
                    vec![src_p],
                ),
                None,
            );
            ddg.output_stream_nodes.push(index);
            ret
        }
    };

    ddg.nodes.push(n);
    if let Some(v) = v {
        ddg.variable_def_nodes[v.0 as usize] = Some(index);
    }
}

fn visit_scalar(s: &program::Scalar, ddg: &Ddg) -> (VariableIndex, NodeIndex) {
    match *s {
        program::Scalar::VariableRef(src) => {
            (src.into(), ddg.variable_def_nodes[src.0 as usize].unwrap())
        }
    }
}

fn schedule(ddg: &Ddg, program: &mut Program) {
    assert!(try_schedule_predecessors(&ddg.output_stream_nodes, ddg, program));
}

fn try_schedule_predecessors(predecessors: &[NodeIndex], ddg: &Ddg, program: &mut Program) -> bool {
    loop {
        let mut has_progressed = false;
        for &p in predecessors {
            if let Some(predecessor_has_progressed) = try_schedule_node(p, ddg, program) {
                has_progressed |= predecessor_has_progressed;
            }
        }
        if !has_progressed {
            return predecessors.iter().all(|&p| ddg.nodes[p.0 as usize].is_scheduled.get());
        }
    }
}

fn try_schedule_node(n: NodeIndex, ddg: &Ddg, program: &mut Program) -> Option<bool> {
    if ddg.nodes[n.0 as usize].is_scheduled.get() {
        return Some(false);
    }
    if !try_schedule_predecessors(&ddg.nodes[n.0 as usize].predecessors, ddg, program) {
        return None;
    }
    program.statements.push(ddg.nodes[n.0 as usize].statement);
    ddg.nodes[n.0 as usize].is_scheduled.set(true);
    Some(true)
}

fn analyze_liveness(p: &Program) -> (BTreeSet<(VariableIndex, VariableIndex)>, Vec<u32>) {
    let mut is_live = BitArray::new(p.variable_registers.len());
    let mut interference_edges = BTreeSet::new();
    let mut variable_degrees = vec![0; p.variable_registers.len()];

    for s in p.statements.iter().rev() {
        let (uses, def) = match *s {
            Statement::Add(dst, lhs, rhs) => (vec![lhs, rhs], Some(dst)),
            Statement::Multiply(dst, lhs, rhs, _) => (vec![lhs, rhs], Some(dst)),
            Statement::Load(dst, _, _) => (Vec::new(), Some(dst)),
            Statement::Store(_, src, _) => (vec![src], None),
        };
        if let Some(d) = def {
            is_live.clear(d.0 as _);
        }
        for u in uses {
            is_live.set(u.0 as _);
        }
        let mut is_live_x_iter = is_live.iter_set_indices();
        while let Some(x) = is_live_x_iter.next().map(|i| VariableIndex(i as _)) {
            let mut is_live_y_iter = is_live_x_iter.clone();
            while let Some(y) = is_live_y_iter.next().map(|i| VariableIndex(i as _)) {
                if interference_edges.insert((x, y)) {
                    variable_degrees[x.0 as usize] += 1;
                    variable_degrees[y.0 as usize] += 1;
                }
            }
        }
    }

    (interference_edges, variable_degrees)
}

fn allocate_registers(
    program: &mut Program,
    interference_edges: BTreeSet<(VariableIndex, VariableIndex)>,
    variable_degrees: Vec<u32>,
) -> Result<(), CompileError> {
    let mut sorted_variables =
        variable_degrees.into_iter().enumerate().map(|(i, d)| (d, VariableIndex(i as _))).collect::<Vec<_>>();
    sorted_variables.sort_by_key(|&(d, _)| d);

    for v in sorted_variables.into_iter().map(|(_, v)| v).rev() {
        let mut register_index = 0;
        while interference_edges.iter().filter_map(|(x, y)| if *x == v {
            Some(*y)
        } else if *y == v {
            Some(*x)
        } else {
            None
        }).any(|other| {
            program
                .reg(other)
                .map(|r| r as u32 == register_index)
                .unwrap_or(false)
        }) {
            register_index += 1;
        }
        let register = Register::from_u32(register_index).ok_or(CompileError::TooManyRegisters)?;
        program.variable_registers[v.0 as usize] = Some(register);
    }

    Ok(())
}

fn generate_instructions(program: &Program) -> Vec<Instruction> {
    program.statements.iter().map(|s| match *s {
        Statement::Add(dst, lhs, rhs) => {
            let dst = program.reg(dst).unwrap();
            let lhs = program.reg(lhs).unwrap();
            let rhs = program.reg(rhs).unwrap();
            add(dst, lhs, rhs)
        }
        Statement::Multiply(dst, lhs, rhs, shift) => {
            let dst = program.reg(dst).unwrap();
            let lhs = program.reg(lhs).unwrap();
            let rhs = program.reg(rhs).unwrap();
            mul(dst, lhs, rhs, shift)
        }
        Statement::Load(dst, src, offset) => {
            let dst = program.reg(dst).unwrap();
            lod(dst, src, offset)
        }
        Statement::Store(dst, src, offset) => {
            let src = program.reg(src).unwrap();
            sto(dst, src, offset)
        }
    }).collect()
}

fn encode_instructions(instructions: &[Instruction]) -> Vec<EncodedInstruction> {
    instructions.iter().map(|&i| i.encode()).collect()
}
