use crate::instructions::*;

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
        pub variables: Vec<String>,
    }

    impl Program {
        pub fn new() -> Program {
            Program {
                statements: Vec::new(),
                variables: Vec::new(),
            }
        }

        pub fn add(&self, lhs: &Expr, rhs: &Expr) -> Expr {
            Expr::Add(Box::new(lhs.clone()), Box::new(rhs.clone()))
        }

        pub fn load(&self, src: InputStream) -> Expr {
            Expr::Load(src)
        }

        pub fn store(&mut self, dst: OutputStream, src: Expr) {
            self.statements.push(Statement::Store(dst, src));
        }

        pub fn var(&mut self, name: impl Into<String>, initializer: Option<&Expr>) -> Expr {
            let name = name.into();
            self.variables.push(name.clone());

            if let Some(initializer) = initializer {
                self.statements.push(Statement::Assign(name.clone(), initializer.clone()));
            }

            Expr::VariableRef(name)
        }
    }

    // TODO: Don't derive Clone/Copy for recursive type!
    #[derive(Clone)]
    pub enum Expr {
        Add(Box<Expr>, Box<Expr>),
        Load(InputStream),
        VariableRef(String),
    }

    pub enum Statement {
        Assign(String, Expr),
        Store(OutputStream, Expr),
    }
}

// TODO: Move?
mod ir {
    use crate::instructions::*;

    #[derive(Debug)]
    pub struct Program {
        pub statements: Vec<Statement>,
        pub variables: Vec<Variable>,
    }

    impl Program {
        pub fn new() -> Program {
            Program {
                statements: Vec::new(),
                variables: Vec::new(),
            }
        }

        pub fn alloc_temp(&mut self) -> String {
            let name = format!("__temp_{}", self.variables.len());
            self.variables.push(Variable::new(name.clone()));
            name
        }
    }

    #[derive(Debug)]
    pub enum Statement {
        Add(String, String, String),
        Assign(String, String),
        Load(String, InputStream),
        Store(OutputStream, String),
    }

    #[derive(Debug)]
    pub struct Variable {
        pub name: String,
        pub register: Option<Register>,
    }

    impl Variable {
        pub fn new(name: String) -> Variable {
            Variable {
                name,
                register: None,
            }
        }
    }
}

pub fn compile(p: &program::Program) -> Result<Vec<Instruction>, CompileError> {
    let mut program = ir::Program::new();

    parse(p, &mut program);
    println!("Parse result: {:#?}", program);

    allocate_registers(&mut program)?;
    println!("Register allocation result: {:#?}", program.variables);

    Ok(generate_instructions(&program))
}

fn parse(p: &program::Program, program: &mut ir::Program) {
    for v in &p.variables {
        // TODO: Validity check(s) etc
        program.variables.push(ir::Variable::new(v.clone()));
    }

    for s in &p.statements {
        parse_statement(s, program);
    }
}

fn parse_statement(s: &program::Statement, program: &mut ir::Program) {
    match *s {
        program::Statement::Assign(ref dst, ref src) => {
            let src = parse_expr(src, program);
            program.statements.push(ir::Statement::Assign(dst.clone(), src));
        }
        program::Statement::Store(dst, ref src) => {
            let src = parse_expr(src, program);
            program.statements.push(ir::Statement::Store(dst, src))
        }
    }
}

fn parse_expr(e: &program::Expr, program: &mut ir::Program) -> String {
    match *e {
        program::Expr::Add(ref lhs, ref rhs) => {
            let lhs = parse_expr(lhs, program);
            let rhs = parse_expr(rhs, program);
            let t = program.alloc_temp();
            program.statements.push(ir::Statement::Add(t.clone(), lhs, rhs));
            t
        }
        program::Expr::Load(src) => {
            let t = program.alloc_temp();
            program.statements.push(ir::Statement::Load(t.clone(), src));
            t
        }
        program::Expr::VariableRef(ref src) => {
            src.clone()
        }
    }
}

fn allocate_registers(program: &mut ir::Program) -> Result<(), CompileError> {
    // TODO: lol :)
    for (i, v) in program.variables.iter_mut().enumerate() {
        v.register = Some(Register::from_u32(i as _).ok_or(CompileError::TooManyRegisters)?);
    }

    Ok(())
}

fn generate_instructions(program: &ir::Program) -> Vec<Instruction> {
    program.statements.iter().map(|s| match *s {
        ir::Statement::Add(ref dst, ref lhs, ref rhs) => {
            // TODO: Factor out variable name -> register mapping
            let dst = program.variables.iter().find(|v| v.name == *dst).unwrap().register.unwrap();
            let lhs = program.variables.iter().find(|v| v.name == *lhs).unwrap().register.unwrap();
            let rhs = program.variables.iter().find(|v| v.name == *rhs).unwrap().register.unwrap();
            add(dst, lhs, rhs)
        }
        ir::Statement::Assign(ref dst, ref src) => {
            let dst = program.variables.iter().find(|v| v.name == *dst).unwrap().register.unwrap();
            let src = program.variables.iter().find(|v| v.name == *src).unwrap().register.unwrap();
            mov(dst, src)
        }
        ir::Statement::Load(ref dst, src) => {
            let dst = program.variables.iter().find(|v| v.name == *dst).unwrap().register.unwrap();
            lod(dst, src)
        }
        ir::Statement::Store(dst, ref src) => {
            let src = program.variables.iter().find(|v| v.name == *src).unwrap().register.unwrap();
            str(dst, src)
        }
    }).collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::instructions::InputStream::*;
    use crate::instructions::OutputStream::*;
    use crate::test_helpers::*;

    #[test]
    fn id() -> Result<(), CompileError> {
        let mut p = program::Program::new();

        let input = I0;
        let output = O0;
        p.store(output, p.load(input));

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream = (0..num_elements).into_iter().collect::<Vec<_>>();
        let expected_output_stream = input_stream.clone();

        test(&instructions, &input_stream, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn x2() -> Result<(), CompileError> {
        let mut p = program::Program::new();

        let input = I0;
        let output = O0;
        let x = p.var("x", Some(&p.load(input)));
        p.store(output, p.add(&x, &x));

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream = (0..num_elements).into_iter().collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().map(|x| x * 2).collect::<Vec<_>>();

        test(&instructions, &input_stream, &expected_output_stream);

        Ok(())
    }
}
