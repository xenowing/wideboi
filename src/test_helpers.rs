use crate::instructions::*;

#[derive(Debug)]
pub struct InputStreamInfo<'a> {
    pub data: &'a [i32],
    pub thread_stride: u32,
}

#[derive(Debug)]
pub struct OutputStreamInfo {
    pub num_words: u32,
    pub thread_stride: u32,
}

fn model<'a>(
    encoded_instructions: &[EncodedInstruction],
    input_stream_infos: &[InputStreamInfo<'a>],
    output_stream_info: OutputStreamInfo,
    num_threads: u32,
) -> (Vec<i32>, u32) {
    let instructions = encoded_instructions.iter().map(|&i| Instruction::decode(i).unwrap()).collect::<Vec<_>>();
    let num_instructions = instructions.len() as u32;
    let mut output_stream = vec![0; output_stream_info.num_words as usize];

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

    let mut contexts = vec![Context::new(vec![0; input_stream_infos.len()]); num_contexts as usize];

    let mut next_context = 0;
    let mut next_thread = 0;

    let mut input_stream_offsets = vec![0; input_stream_infos.len()];
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
                Instruction::Load(dst, src) => {
                    let input_stream_index = match src {
                        InputStream::I0 => 0,
                        InputStream::I1 => 1,
                    };
                    context.registers[dst as usize] = input_stream_infos[input_stream_index].data[context.input_stream_offsets[input_stream_index] as usize];
                    context.input_stream_offsets[input_stream_index] += 1;
                }
                Instruction::Multiply(dst, lhs, rhs) => {
                    let lhs = context.registers[lhs as usize];
                    let rhs = context.registers[rhs as usize];
                    context.registers[dst as usize] = lhs.wrapping_mul(rhs);
                }
                Instruction::Store(_dst, src) => {
                    output_stream[context.output_stream_offset as usize] = context.registers[src as usize];
                    context.output_stream_offset += 1;
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
            for (offset, input_stream_info) in input_stream_offsets.iter_mut().zip(input_stream_infos.iter()) {
                *offset += input_stream_info.thread_stride;
            }
            output_stream_offset += output_stream_info.thread_stride;
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

pub fn test<'a>(
    encoded_instructions: &[EncodedInstruction],
    input_stream_infos: &[InputStreamInfo<'a>],
    output_stream_info: OutputStreamInfo,
    num_threads: u32,
    expected_output_stream: &[i32],
) {
    println!("input stream infos: {:#?}", input_stream_infos);
    println!("output stream info: {:#?}", output_stream_info);
    println!("expected output stream: {:?}", expected_output_stream);

    println!("testing model");

    let (output_stream, num_cycles) = model(&encoded_instructions, input_stream_infos, output_stream_info, num_threads);
    println!(" - output stream: {:?}", output_stream);
    println!(" - num cycles: {}", num_cycles);

    assert_eq!(output_stream, expected_output_stream);

    println!("success!");
}
