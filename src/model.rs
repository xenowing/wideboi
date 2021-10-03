use crate::compiler::*;
use crate::instructions::*;

pub fn model(
    program: &CompiledProgram,
    input_stream_bindings: &[&[i32]],
    uniform_data: &[i32],
    num_threads: u32,
) -> (Vec<Vec<i32>>, u32) {
    let instructions = program.instructions.iter().map(|&i| Instruction::decode(i).unwrap()).collect::<Vec<_>>();
    let num_instructions = instructions.len() as u32;
    let mut output_streams = program.output_stream_thread_strides.iter().map(|&stride| vec![0; (num_threads * stride) as usize]).collect::<Vec<_>>();

    let num_contexts = 8;

    #[derive(Clone)]
    struct Context {
        current_instruction: Option<Instruction>,

        pc: u32,
        registers: [i32; Register::VARIANT_COUNT],
        input_stream_offsets: Vec<u32>,
        output_stream_offsets: Vec<u32>,
    }

    impl Context {
        fn new(input_stream_offsets: Vec<u32>, output_stream_offsets: Vec<u32>) -> Context {
            Context {
                current_instruction: None,

                pc: 0,
                registers: [0; Register::VARIANT_COUNT],
                input_stream_offsets,
                output_stream_offsets,
            }
        }
    }

    let mut input_stream_offsets = vec![0; program.input_stream_thread_strides.len()];
    let mut output_stream_offsets = vec![0; program.output_stream_thread_strides.len()];

    let mut contexts = vec![Context::new(input_stream_offsets.clone(), output_stream_offsets.clone()); num_contexts as usize];

    let mut next_context = 0;
    let mut next_thread = 0;

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
                Instruction::Load(dst, src, offset) => {
                    let src = src as usize;
                    context.registers[dst as usize] = input_stream_bindings[src][context.input_stream_offsets[src] as usize + offset as usize];
                }
                Instruction::Multiply(dst, lhs, rhs, shift) => {
                    let lhs = context.registers[lhs as usize] as i64;
                    let rhs = context.registers[rhs as usize] as i64;
                    context.registers[dst as usize] = ((lhs * rhs) >> shift) as _;
                }
                Instruction::PositivePart(dst, src) => {
                    let src = context.registers[src as usize];
                    context.registers[dst as usize] = if src.is_positive() { src } else { 0 };
                }
                Instruction::Store(dst, src, offset) => {
                    let dst = dst as usize;
                    let src = context.registers[src as usize];
                    output_streams[dst][context.output_stream_offsets[dst] as usize + offset as usize] = src;
                }
                Instruction::Uniform(dst, offset) => {
                    context.registers[dst as usize] = uniform_data[offset as usize];
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
            context.input_stream_offsets = input_stream_offsets.clone();
            context.output_stream_offsets = output_stream_offsets.clone();
            next_thread += 1;
            for (offset, stride) in input_stream_offsets.iter_mut().zip(program.input_stream_thread_strides.iter()) {
                *offset += stride;
            }
            for (offset, stride) in output_stream_offsets.iter_mut().zip(program.output_stream_thread_strides.iter()) {
                *offset += stride;
            }
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

    (output_streams, num_cycles)
}
