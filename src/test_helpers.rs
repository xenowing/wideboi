use crate::compiler::*;
use crate::instructions::*;

fn model(
    program: &CompiledProgram,
    input_stream_bindings: &[&[i32]],
    num_threads: u32,
) -> (Vec<i32>, u32) {
    let instructions = program.instructions.iter().map(|&i| Instruction::decode(i).unwrap()).collect::<Vec<_>>();
    let num_instructions = instructions.len() as u32;
    let mut output_stream = vec![0; (num_threads * program.output_stream_thread_stride) as usize];

    let num_contexts = 8;

    #[derive(Clone)]
    struct Context {
        current_instruction: Option<Instruction>,

        pc: u32,
        registers: [i32; Register::VARIANT_COUNT],
        input_stream_offsets: Vec<u32>,
        output_stream_offset: u32,
    }

    impl Context {
        fn new(input_stream_offsets: Vec<u32>) -> Context {
            Context {
                current_instruction: None,

                pc: 0,
                registers: [0; Register::VARIANT_COUNT],
                input_stream_offsets,
                output_stream_offset: 0,
            }
        }
    }

    let mut contexts = vec![Context::new(vec![0; program.input_stream_thread_strides.len()]); num_contexts as usize];

    let mut next_context = 0;
    let mut next_thread = 0;

    let mut input_stream_offsets = vec![0; program.input_stream_thread_strides.len()];
    let mut output_stream_offset = 0;

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
                    let input_stream_index = match src {
                        InputStream::I0 => 0,
                        InputStream::I1 => 1,
                    };
                    context.registers[dst as usize] = input_stream_bindings[input_stream_index][context.input_stream_offsets[input_stream_index] as usize + offset as usize];
                }
                Instruction::Multiply(dst, lhs, rhs, shift) => {
                    let lhs = context.registers[lhs as usize] as i64;
                    let rhs = context.registers[rhs as usize] as i64;
                    context.registers[dst as usize] = (lhs.wrapping_mul(rhs) >> shift) as _;
                }
                Instruction::Store(_dst, src, offset) => {
                    output_stream[context.output_stream_offset as usize + offset as usize] = context.registers[src as usize];
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
            context.output_stream_offset = output_stream_offset;
            next_thread += 1;
            for (offset, stride) in input_stream_offsets.iter_mut().zip(program.input_stream_thread_strides.iter()) {
                *offset += stride;
            }
            output_stream_offset += program.output_stream_thread_stride;
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

pub fn test(
    program: &CompiledProgram,
    input_stream_bindings: &[&[i32]],
    num_threads: u32,
    expected_output_stream: &[i32],
) {
    println!("program: {:#?}", program);
    println!("expected output stream: {:?}", expected_output_stream);

    for (&binding, &stride) in input_stream_bindings.iter().zip(program.input_stream_thread_strides.iter()) {
        assert_eq!(binding.len() as u32, num_threads * stride);
    }

    println!("testing model");

    let (output_stream, num_cycles) = model(program, input_stream_bindings, num_threads);
    println!(" - output stream: {:?}", output_stream);
    println!(" - num cycles: {}", num_cycles);

    assert_eq!(output_stream, expected_output_stream);

    println!("success!");
}
