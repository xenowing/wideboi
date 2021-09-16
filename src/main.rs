use std::fmt;

#[derive(Clone, Copy, Debug)]
enum InputStream {
    I0,
}

impl fmt::Display for InputStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Copy, Debug)]
enum OutputStream {
    O0,
}

impl fmt::Display for OutputStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[allow(unused)] // TODO: Remove?
#[derive(Clone, Copy, Debug)]
enum Register {
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Copy, Debug)]
enum Instruction {
    Add(Register, Register, Register),
    Load(Register, InputStream),
    Store(OutputStream, Register),
}

impl Instruction {
    fn mnemonic(&self) -> &'static str {
        match self {
            Instruction::Add(_, _, _) => "add",
            Instruction::Load(_, _) => "lod",
            Instruction::Store(_, _) => "str",
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mnemonic = self.mnemonic();
        match self {
            Instruction::Add(dst, lhs, rhs) => write!(f, "{} {}, {}, {}", mnemonic, dst, lhs, rhs),
            Instruction::Load(dst, src) => write!(f, "{} {}, {}", mnemonic, dst, src),
            Instruction::Store(dst, src) => write!(f, "{} {}, {}", mnemonic, dst, src),
        }
    }
}

fn add(dst: Register, lhs: Register, rhs: Register) -> Instruction {
    Instruction::Add(dst, lhs, rhs)
}

fn lod(dst: Register, src: InputStream) -> Instruction {
    Instruction::Load(dst, src)
}

fn str(dst: OutputStream, src: Register) -> Instruction {
    Instruction::Store(dst, src)
}

fn model(instructions: &[Instruction], input_stream: &[u32]) -> (Vec<u32>, u32) {
    let num_instructions = instructions.len() as u32;
    let num_threads = input_stream.len() as u32;
    let mut output_stream = Vec::new();

    let num_contexts = 8;

    #[derive(Clone)]
    struct Context {
        current_instruction: Option<Instruction>,

        pc: u32,
        registers: [u32; 8],
    }

    impl Context {
        fn new() -> Context {
            Context {
                current_instruction: None,

                pc: 0,
                registers: [0; 8],
            }
        }
    }

    let mut contexts = vec![Context::new(); num_contexts as usize];

    let mut next_context = 0;
    let mut next_thread = 0;

    let mut input_stream_offset = 0;

    let mut num_cycles = 0;

    loop {
        let context = &mut contexts[next_context as usize];
        if let Some(instruction) = context.current_instruction {
            match instruction {
                Instruction::Add(dst, lhs, rhs) => {
                    let lhs = context.registers[lhs as usize];
                    let rhs = context.registers[rhs as usize];
                    context.registers[dst as usize] = lhs.wrapping_add(rhs);
                }
                Instruction::Load(dst, _src) => {
                    context.registers[dst as usize] = input_stream[input_stream_offset];
                    input_stream_offset += 1;
                }
                Instruction::Store(_dst, src) => {
                    output_stream.push(context.registers[src as usize]);
                }
            }

            context.pc += 1;

            context.current_instruction = if context.pc < num_instructions {
                Some(instructions[context.pc as usize])
            } else {
                None
            };
        }

        if context.current_instruction.is_none() && next_thread < num_threads {
            context.pc = 0;
            context.current_instruction = Some(instructions[context.pc as usize]);
            next_thread += 1;
        }

        print!("cycle {:04}", num_cycles);
        for context in &contexts {
            print!(" | ");
            if let Some(instruction) = context.current_instruction {
                print!("{}", instruction.mnemonic());
            } else {
                print!("   ");
            }
        }
        println!(" |");

        if next_thread == num_threads && contexts.iter().map(|c| c.current_instruction.is_none()).all(|x| x) {
            break;
        }

        next_context = (next_context + 1) % num_contexts;
        num_cycles += 1;
    }

    (output_stream, num_cycles)
}

fn main() {
    use InputStream::*;
    use OutputStream::*;
    use Register::*;

    let instructions = vec![
        lod(R0, I0),
        add(R0, R0, R0),
        str(O0, R0),
    ];
    println!("instructions:");
    for instruction in &instructions {
        println!("  {}", instruction);
    }

    let num_threads = 10;

    let input_stream = (0..num_threads).into_iter().collect::<Vec<_>>();
    println!("input stream: {:?}", input_stream);

    let (output_stream, num_cycles) = model(&instructions, &input_stream);
    println!("num cycles: {}", num_cycles);

    assert_eq!(output_stream, (0..num_threads).into_iter().map(|x| x * 2).collect::<Vec<_>>());

    println!("success!");
}
