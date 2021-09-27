use crate::bit_vector::*;
use crate::instructions::*;

use std::collections::{BTreeSet, HashMap};

use self::ir::NodeIndex;
use self::ir::VariableIndex;

#[derive(Debug)]
pub enum CompileError {
    // TODO: Proper desc/fields
    // TODO: Test(s)
    TooManyRegisters,
}

// TODO: Move?
mod program {
    use crate::instructions::*;

    pub struct Program {
        pub statements: Vec<Statement>,
        pub num_variables: u32,
    }

    impl Program {
        pub fn new() -> Program {
            Program {
                statements: Vec::new(),
                num_variables: 0,
            }
        }

        pub fn add_s(&mut self, lhs: Scalar, rhs: Scalar) -> Scalar {
            let t = self.alloc_var();
            self.statements.push(Statement::Add(t, lhs, rhs));
            Scalar::VariableRef(t)
        }

        pub fn add_v3(&mut self, lhs: V3, rhs: V3) -> V3 {
            V3 {
                x: self.add_s(lhs.x, rhs.x),
                y: self.add_s(lhs.y, rhs.y),
                z: self.add_s(lhs.z, rhs.z),
            }
        }

        fn alloc_var(&mut self) -> VariableIndex {
            let ret = VariableIndex(self.num_variables);
            self.num_variables += 1;
            ret
        }

        pub fn dot(&mut self, lhs: V3, rhs: V3, shift: u8) -> Scalar {
            let temp = self.mul_v3(lhs, rhs, shift);
            let lhs = self.add_s(temp.x, temp.y);
            self.add_s(lhs, temp.z)
        }

        pub fn load_s(&mut self, src: InputStream) -> Scalar {
            let name = self.alloc_var();
            self.statements.push(Statement::Load(name, src));
            Scalar::VariableRef(name)
        }

        pub fn load_v3(&mut self, src: InputStream) -> V3 {
            V3 {
                x: self.load_s(src),
                y: self.load_s(src),
                z: self.load_s(src),
            }
        }

        pub fn mul_s(&mut self, lhs: Scalar, rhs: Scalar, shift: u8) -> Scalar {
            let t = self.alloc_var();
            self.statements.push(Statement::Multiply(t, lhs, rhs, shift & SHIFT_FIELD_MASK));
            Scalar::VariableRef(t)
        }

        pub fn mul_v3(&mut self, lhs: V3, rhs: V3, shift: u8) -> V3 {
            V3 {
                x: self.mul_s(lhs.x, rhs.x, shift),
                y: self.mul_s(lhs.y, rhs.y, shift),
                z: self.mul_s(lhs.z, rhs.z, shift),
            }
        }

        pub fn store_s(&mut self, dst: OutputStream, src: Scalar) {
            self.statements.push(Statement::Store(dst, src));
        }

        pub fn store_v3(&mut self, dst: OutputStream, src: V3) {
            self.store_s(dst, src.x);
            self.store_s(dst, src.y);
            self.store_s(dst, src.z);
        }
    }

    #[derive(Clone)]
    pub enum Scalar {
        VariableRef(VariableIndex),
    }

    pub enum Statement {
        Add(VariableIndex, Scalar, Scalar),
        Load(VariableIndex, InputStream),
        Multiply(VariableIndex, Scalar, Scalar, u8),
        Store(OutputStream, Scalar),
    }

    #[derive(Clone)]
    pub struct V3 {
        pub x: Scalar,
        pub y: Scalar,
        pub z: Scalar,
    }

    #[derive(Clone, Copy)]
    pub struct VariableIndex(pub u32);
}

// TODO: Move?
mod ir {
    use super::*;

    use std::cell::Cell;

    #[derive(Debug)]
    pub struct Program {
        pub statements: Vec<Statement>,
        pub variable_registers: Vec<Option<Register>>,
    }

    impl Program {
        pub fn new(num_variables: u32) -> Program {
            Program {
                statements: Vec::new(),
                variable_registers: vec![None; num_variables as usize],
            }
        }

        pub fn reg(&self, variable_index: VariableIndex) -> Option<Register> {
            self.variable_registers[variable_index.0 as usize]
        }
    }

    #[derive(Debug)]
    pub struct Ddg {
        pub nodes: Vec<Node>,
        pub variable_def_nodes: Vec<Option<NodeIndex>>,
        pub previous_input_stream_nodes: HashMap<InputStream, NodeIndex>,
        pub output_stream_nodes: Vec<NodeIndex>,
    }

    impl Ddg {
        pub fn new(num_variables: u32) -> Ddg {
            Ddg {
                nodes: Vec::new(),
                variable_def_nodes: vec![None; num_variables as usize],
                previous_input_stream_nodes: HashMap::new(),
                output_stream_nodes: Vec::new(),
            }
        }
    }

    #[derive(Debug)]
    pub struct Node {
        pub statement: Statement,
        pub predecessors: Vec<NodeIndex>,
        pub is_scheduled: Cell<bool>,
    }

    impl Node {
        pub fn new(statement: Statement, predecessors: Vec<NodeIndex>) -> Node {
            Node {
                statement,
                predecessors,
                is_scheduled: Cell::new(false),
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct NodeIndex(pub u32);

    #[derive(Debug, Clone, Copy)]
    pub enum Statement {
        Add(VariableIndex, VariableIndex, VariableIndex),
        Load(VariableIndex, InputStream),
        Multiply(VariableIndex, VariableIndex, VariableIndex, u8),
        Store(OutputStream, VariableIndex),
    }

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    pub struct VariableIndex(pub u32);

    impl From<program::VariableIndex> for VariableIndex {
        fn from(i: program::VariableIndex) -> Self {
            Self(i.0)
        }
    }
}

pub fn compile(p: &program::Program) -> Result<Vec<EncodedInstruction>, CompileError> {
    let ddg = construct_ddg(p);
    println!("Directed dependency graph: {:#?}", ddg);

    let mut program = ir::Program::new(p.num_variables);

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

    Ok(encoded_instructions)
}

fn construct_ddg(p: &program::Program) -> ir::Ddg {
    let mut ddg = ir::Ddg::new(p.num_variables);

    for s in &p.statements {
        visit_statement(s, &mut ddg);
    }

    ddg
}

fn visit_statement(s: &program::Statement, ddg: &mut ir::Ddg) {
    let index = ir::NodeIndex(ddg.nodes.len() as _);

    let (n, v) = match *s {
        program::Statement::Add(dst, ref lhs, ref rhs) => {
            let (lhs, lhs_p) = visit_scalar(lhs, ddg);
            let (rhs, rhs_p) = visit_scalar(rhs, ddg);
            (
                ir::Node::new(
                    ir::Statement::Add(dst.into(), lhs, rhs),
                    vec![lhs_p, rhs_p],
                ),
                Some(dst),
            )
        }
        program::Statement::Load(dst, src) => {
            (
                ir::Node::new(
                    ir::Statement::Load(dst.into(), src),
                    if let Some(p) = ddg.previous_input_stream_nodes.insert(src, index) {
                        vec![p]
                    } else {
                        Vec::new()
                    },
                ),
                Some(dst),
            )
        }
        program::Statement::Multiply(dst, ref lhs, ref rhs, shift) => {
            let (lhs, lhs_p) = visit_scalar(lhs, ddg);
            let (rhs, rhs_p) = visit_scalar(rhs, ddg);
            (
                ir::Node::new(
                    ir::Statement::Multiply(dst.into(), lhs, rhs, shift),
                    vec![lhs_p, rhs_p],
                ),
                Some(dst)
            )
        }
        program::Statement::Store(dst, ref src) => {
            let (src, src_p) = visit_scalar(src, ddg);
            let ret = (
                ir::Node::new(
                    ir::Statement::Store(dst, src),
                    if let Some(&p) = ddg.output_stream_nodes.last() {
                        vec![src_p, p]
                    } else {
                        vec![src_p]
                    },
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

fn visit_scalar(s: &program::Scalar, ddg: &ir::Ddg) -> (ir::VariableIndex, NodeIndex) {
    match *s {
        program::Scalar::VariableRef(src) => {
            (src.into(), ddg.variable_def_nodes[src.0 as usize].unwrap())
        }
    }
}

fn schedule(ddg: &ir::Ddg, program: &mut ir::Program) {
    assert!(try_schedule_predecessors(&ddg.output_stream_nodes, ddg, program));
}

fn try_schedule_predecessors(predecessors: &[NodeIndex], ddg: &ir::Ddg, program: &mut ir::Program) -> bool {
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

fn try_schedule_node(n: NodeIndex, ddg: &ir::Ddg, program: &mut ir::Program) -> Option<bool> {
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

fn analyze_liveness(p: &ir::Program) -> (BTreeSet<(ir::VariableIndex, ir::VariableIndex)>, Vec<u32>) {
    let mut is_live = BitVector::new(p.variable_registers.len());
    let mut interference_edges = BTreeSet::new();
    let mut variable_degrees = vec![0; p.variable_registers.len()];

    for s in p.statements.iter().rev() {
        let (uses, def) = match *s {
            ir::Statement::Add(dst, lhs, rhs) => (vec![lhs, rhs], Some(dst)),
            ir::Statement::Multiply(dst, lhs, rhs, _) => (vec![lhs, rhs], Some(dst)),
            ir::Statement::Load(dst, _) => (Vec::new(), Some(dst)),
            ir::Statement::Store(_, src) => (vec![src], None),
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
    program: &mut ir::Program,
    interference_edges: BTreeSet<(ir::VariableIndex, ir::VariableIndex)>,
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

fn generate_instructions(program: &ir::Program) -> Vec<Instruction> {
    program.statements.iter().map(|s| match *s {
        ir::Statement::Add(dst, lhs, rhs) => {
            let dst = program.reg(dst).unwrap();
            let lhs = program.reg(lhs).unwrap();
            let rhs = program.reg(rhs).unwrap();
            add(dst, lhs, rhs)
        }
        ir::Statement::Multiply(dst, lhs, rhs, shift) => {
            let dst = program.reg(dst).unwrap();
            let lhs = program.reg(lhs).unwrap();
            let rhs = program.reg(rhs).unwrap();
            mul(dst, lhs, rhs, shift)
        }
        ir::Statement::Load(dst, src) => {
            let dst = program.reg(dst).unwrap();
            lod(dst, src)
        }
        ir::Statement::Store(dst, src) => {
            let src = program.reg(src).unwrap();
            sto(dst, src)
        }
    }).collect()
}

fn encode_instructions(instructions: &[Instruction]) -> Vec<EncodedInstruction> {
    instructions.iter().map(|&i| i.encode()).collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compiler::program::Program;
    use crate::instructions::InputStream::*;
    use crate::instructions::OutputStream::*;
    use crate::test_helpers::*;

    #[test]
    fn id() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input = I0;
        let output = O0;
        let x = p.load_s(input);
        p.store_s(output, x);

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream = (0..num_elements).into_iter().collect::<Vec<_>>();
        let expected_output_stream = input_stream.clone();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream,
                thread_stride: 1,
            },
        ], OutputStreamInfo {
            num_words: num_elements as _,
            thread_stride: 1,
        }, num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn x2() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input = I0;
        let output = O0;
        let x = p.load_s(input);
        let res = p.add_s(x.clone(), x);
        p.store_s(output, res);

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream = (0..num_elements).into_iter().collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().map(|x| x * 2).collect::<Vec<_>>();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream,
                thread_stride: 1,
            },
        ], OutputStreamInfo {
            num_words: num_elements as _,
            thread_stride: 1,
        }, num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn scalar_sums() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input_x = I0;
        let input_y = I1;
        let output = O0;
        let x = p.load_s(input_x);
        let y = p.load_s(input_y);
        let res = p.add_s(x, y);
        p.store_s(output, res);

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream_x = (0..num_elements).into_iter().collect::<Vec<_>>();
        let input_stream_y = (0..num_elements).into_iter().map(|x| x * 2).collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().map(|x| x * 3).collect::<Vec<_>>();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream_x,
                thread_stride: 1,
            },
            InputStreamInfo {
                data: &input_stream_y,
                thread_stride: 1,
            },
        ], OutputStreamInfo {
            num_words: num_elements as _,
            thread_stride: 1,
        }, num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn scalar_muls_integer() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input_x = I0;
        let input_y = I1;
        let output = O0;
        let x = p.load_s(input_x);
        let y = p.load_s(input_y);
        let res = p.mul_s(x, y, 0);
        p.store_s(output, res);

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream_x = (0..num_elements).into_iter().collect::<Vec<_>>();
        let input_stream_y = (0..num_elements).into_iter().map(|x| x * 2).collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().map(|x| x * x * 2).collect::<Vec<_>>();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream_x,
                thread_stride: 1,
            },
            InputStreamInfo {
                data: &input_stream_y,
                thread_stride: 1,
            },
        ], OutputStreamInfo {
            num_words: num_elements as _,
            thread_stride: 1,
        }, num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn vector_sums() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input_x = I0;
        let input_y = I1;
        let output = O0;
        let x = p.load_v3(input_x);
        let y = p.load_v3(input_y);
        let res = p.add_v3(x, y);
        p.store_v3(output, res);

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream_x = (0..num_elements).into_iter().flat_map(|x| [x * 1 + 0, x * 1 + 1, x * 1 + 2]).collect::<Vec<_>>();
        let input_stream_y = (0..num_elements).into_iter().flat_map(|x| [x * 2 + 0, x * 2 + 1, x * 2 + 2]).collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().flat_map(|x| [x * 3 + 0, x * 3 + 2, x * 3 + 4]).collect::<Vec<_>>();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream_x,
                thread_stride: 3,
            },
            InputStreamInfo {
                data: &input_stream_y,
                thread_stride: 3,
            },
        ], OutputStreamInfo {
            num_words: (num_elements * 3) as _,
            thread_stride: 3,
        }, num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn vector_muls_integer() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input_x = I0;
        let input_y = I1;
        let output = O0;
        let x = p.load_v3(input_x);
        let y = p.load_v3(input_y);
        let res = p.mul_v3(x, y, 0);
        p.store_v3(output, res);

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream_x = (0..num_elements).into_iter().flat_map(|x| [x * 1 + 0, x * 1 + 1, x * 1 + 2]).collect::<Vec<_>>();
        let input_stream_y = (0..num_elements).into_iter().flat_map(|x| [x * 2 + 0, x * 2 + 1, x * 2 + 2]).collect::<Vec<_>>();
        let expected_output_stream =
            input_stream_x.iter()
            .zip(input_stream_y.iter())
            .map(|(&x, &y)| x * y)
            .collect::<Vec<_>>();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream_x,
                thread_stride: 3,
            },
            InputStreamInfo {
                data: &input_stream_y,
                thread_stride: 3,
            },
        ], OutputStreamInfo {
            num_words: (num_elements * 3) as _,
            thread_stride: 3,
        }, num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn vector_dots_integer() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input_x = I0;
        let input_y = I1;
        let output = O0;
        let x = p.load_v3(input_x);
        let y = p.load_v3(input_y);
        let res = p.dot(x, y, 0);
        p.store_s(output, res);

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream_x_v3 = (0..num_elements).into_iter().map(|x| [x * 1 + 0, x * 1 + 1, x * 1 + 2]).collect::<Vec<_>>();
        let input_stream_y_v3 = (0..num_elements).into_iter().map(|x| [x * 2 + 0, x * 2 + 1, x * 2 + 2]).collect::<Vec<_>>();
        let expected_output_stream =
            input_stream_x_v3.iter()
            .zip(input_stream_y_v3.iter())
            .map(|(x, y)| (x[0] * y[0]) + (x[1] * y[1]) + (x[2] * y[2]))
            .collect::<Vec<_>>();
        let input_stream_x = input_stream_x_v3.into_iter().flat_map(|x| x).collect::<Vec<_>>();
        let input_stream_y = input_stream_y_v3.into_iter().flat_map(|x| x).collect::<Vec<_>>();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream_x,
                thread_stride: 3,
            },
            InputStreamInfo {
                data: &input_stream_y,
                thread_stride: 3,
            },
        ], OutputStreamInfo {
            num_words: num_elements as _,
            thread_stride: 1,
        }, num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn vector_dots_q() -> Result<(), CompileError> {
        let mut p = Program::new();

        let q_shift = 16;

        let input_x = I0;
        let input_y = I1;
        let output = O0;
        let x = p.load_v3(input_x);
        let y = p.load_v3(input_y);
        let res = p.dot(x, y, q_shift);
        p.store_s(output, res);

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream_x_v3 = (0..num_elements).into_iter().map(|x| [
            (x * 1 + 0) << q_shift,
            (x * 1 + 1) << q_shift,
            (x * 1 + 2) << q_shift,
        ]).collect::<Vec<_>>();
        let input_stream_y_v3 = (0..num_elements).into_iter().map(|x| [
            (x * 2 + 0) << q_shift,
            (x * 2 + 1) << q_shift,
            (x * 2 + 2) << q_shift,
        ]).collect::<Vec<_>>();
        let expected_output_stream =
            input_stream_x_v3.iter()
            .zip(input_stream_y_v3.iter())
            .map(|(x, y)|
                (((x[0] as i64 * y[0] as i64) >> q_shift) as i32) +
                (((x[1] as i64 * y[1] as i64) >> q_shift) as i32) +
                (((x[2] as i64 * y[2] as i64) >> q_shift) as i32)
            )
            .collect::<Vec<_>>();
        let input_stream_x = input_stream_x_v3.into_iter().flat_map(|x| x).collect::<Vec<_>>();
        let input_stream_y = input_stream_y_v3.into_iter().flat_map(|x| x).collect::<Vec<_>>();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream_x,
                thread_stride: 3,
            },
            InputStreamInfo {
                data: &input_stream_y,
                thread_stride: 3,
            },
        ], OutputStreamInfo {
            num_words: num_elements as _,
            thread_stride: 1,
        }, num_elements as _, &expected_output_stream);

        Ok(())
    }
}
