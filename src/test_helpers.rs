use crate::instructions::*;

fn model(instructions: &[Instruction], input_streams: &[&[i32]], num_threads: u32) -> (Vec<i32>, u32) {
    let num_instructions = instructions.len() as u32;
    let mut output_stream = Vec::new();

    let num_contexts = 8;

    #[derive(Clone)]
    struct Context {
        current_instruction: Option<Instruction>,

        pc: u32,
        registers: [i32; 16],
    }

    impl Context {
        fn new() -> Context {
            Context {
                current_instruction: None,

                pc: 0,
                registers: [0; 16],
            }
        }
    }

    let mut contexts = vec![Context::new(); num_contexts as usize];

    let mut next_context = 0;
    let mut next_thread = 0;

    let mut input_stream_offsets = vec![0; input_streams.len()];

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
                Instruction::Load(dst, src) => {
                    let input_stream_index = match src {
                        InputStream::I0 => 0,
                        InputStream::I1 => 1,
                    };
                    context.registers[dst as usize] = input_streams[input_stream_index][input_stream_offsets[input_stream_index]];
                    input_stream_offsets[input_stream_index] += 1;
                }
                Instruction::Mul(dst, lhs, rhs) => {
                    let lhs = context.registers[lhs as usize];
                    let rhs = context.registers[rhs as usize];
                    context.registers[dst as usize] = lhs.wrapping_mul(rhs);
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

pub fn test(instructions: &[Instruction], input_streams: &[&[i32]], num_threads: u32, expected_output_stream: &[i32]) {
    println!("instructions:");
    for instruction in instructions {
        println!("  {}", instruction);
    }

    println!("input streams: {:?}", input_streams);
    println!("expected output stream: {:?}", expected_output_stream);

    println!("testing model");

    let (output_stream, num_cycles) = model(&instructions, input_streams, num_threads);
    println!(" - output stream: {:?}", output_stream);
    println!(" - num cycles: {}", num_cycles);

    assert_eq!(output_stream, expected_output_stream);

    println!("success!");
}
