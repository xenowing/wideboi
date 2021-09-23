use variant_count::*;

use std::fmt;
use std::mem::size_of;

#[derive(Clone, Copy, Debug, Eq, PartialEq, VariantCount)]
pub enum InputStream {
    I0,
    I1,
}

impl InputStream {
    // TODO: Test(s)
    pub fn from_u32(x: u32) -> Option<InputStream> {
        match x {
            0 => Some(InputStream::I0),
            1 => Some(InputStream::I1),
            _ => None
        }
    }
}

impl fmt::Display for InputStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, VariantCount)]
pub enum OutputStream {
    O0,
}

impl OutputStream {
    // TODO: Test(s)
    pub fn from_u32(x: u32) -> Option<OutputStream> {
        match x {
            0 => Some(OutputStream::O0),
            _ => None
        }
    }
}

impl fmt::Display for OutputStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, VariantCount)]
pub enum Register {
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

impl Register {
    // TODO: Test(s)
    pub fn from_u32(x: u32) -> Option<Register> {
        match x {
            0 => Some(Register::R0),
            1 => Some(Register::R1),
            2 => Some(Register::R2),
            3 => Some(Register::R3),
            4 => Some(Register::R4),
            5 => Some(Register::R5),
            6 => Some(Register::R6),
            7 => Some(Register::R7),
            8 => Some(Register::R8),
            9 => Some(Register::R9),
            10 => Some(Register::R10),
            11 => Some(Register::R11),
            12 => Some(Register::R12),
            13 => Some(Register::R13),
            14 => Some(Register::R14),
            15 => Some(Register::R15),
            _ => None
        }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

const_assert!(Register::VARIANT_COUNT.is_power_of_two());

#[derive(Clone, Copy, Debug, Eq, PartialEq, VariantCount)]
pub enum Instruction {
    Add(Register, Register, Register),
    Load(Register, InputStream),
    Multiply(Register, Register, Register),
    Store(OutputStream, Register),
}

impl Instruction {
    pub fn encode(&self) -> EncodedInstruction {
        let e = Encoding::new();

        let opcode = self.opcode();

        match *self {
            Instruction::Add(dst, lhs, rhs) => {
                0
                .set(&e.opcode, opcode as _)
                .set(&e.dst_reg, dst as _)
                .set(&e.lhs_reg, lhs as _)
                .set(&e.rhs_reg, rhs as _)
            }
            Instruction::Load(dst, src) => {
                0
                .set(&e.opcode, opcode as _)
                .set(&e.dst_reg, dst as _)
                .set(&e.src_input_stream, src as _)
            }
            Instruction::Multiply(dst, lhs, rhs) => {
                0
                .set(&e.opcode, opcode as _)
                .set(&e.dst_reg, dst as _)
                .set(&e.lhs_reg, lhs as _)
                .set(&e.rhs_reg, rhs as _)
            }
            Instruction::Store(dst, src) => {
                0
                .set(&e.opcode, opcode as _)
                .set(&e.dst_output_stream, dst as _)
                .set(&e.src_reg, src as _)
            }
        }
    }

    pub fn decode(i: EncodedInstruction) -> Option<Instruction> {
        let e = Encoding::new();

        let opcode = Opcode::from_u32(i.value(&e.opcode));
        let dst_reg = Register::from_u32(i.value(&e.dst_reg));
        let dst_output_stream = OutputStream::from_u32(i.value(&e.dst_output_stream));
        let src_reg = Register::from_u32(i.value(&e.src_reg));
        let src_input_stream = InputStream::from_u32(i.value(&e.src_input_stream));
        let lhs_reg = Register::from_u32(i.value(&e.lhs_reg));
        let rhs_reg = Register::from_u32(i.value(&e.rhs_reg));

        match opcode? {
            Opcode::Add => Some(Instruction::Add(dst_reg?, lhs_reg?, rhs_reg?)),
            Opcode::Load => Some(Instruction::Load(dst_reg?, src_input_stream?)),
            Opcode::Multiply => Some(Instruction::Multiply(dst_reg?, lhs_reg?, rhs_reg?)),
            Opcode::Store => Some(Instruction::Store(dst_output_stream?, src_reg?)),
        }
    }

    // TODO: Test(s)
    pub fn mnemonic(&self) -> &'static str {
        match self {
            Instruction::Add(_, _, _) => "add",
            Instruction::Load(_, _) => "lod",
            Instruction::Multiply(_, _, _) => "mul",
            Instruction::Store(_, _) => "sto",
        }
    }

    // TODO: Test(s)
    fn opcode(&self) -> Opcode {
        match self {
            Instruction::Add(_, _, _) => Opcode::Add,
            Instruction::Load(_, _) => Opcode::Load,
            Instruction::Multiply(_, _, _) => Opcode::Multiply,
            Instruction::Store(_, _) => Opcode::Store,
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mnemonic = self.mnemonic();
        match *self {
            Instruction::Add(dst, lhs, rhs) => write!(f, "{} {}, {}, {}", mnemonic, dst, lhs, rhs),
            Instruction::Load(dst, src) => write!(f, "{} {}, {}", mnemonic, dst, src),
            Instruction::Multiply(dst, lhs, rhs) => write!(f, "{} {}, {}, {}", mnemonic, dst, lhs, rhs),
            Instruction::Store(dst, src) => write!(f, "{} {}, {}", mnemonic, dst, src),
        }
    }
}

pub fn add(dst: Register, lhs: Register, rhs: Register) -> Instruction {
    Instruction::Add(dst, lhs, rhs)
}

pub fn lod(dst: Register, src: InputStream) -> Instruction {
    Instruction::Load(dst, src)
}

pub fn mul(dst: Register, lhs: Register, rhs: Register) -> Instruction {
    Instruction::Multiply(dst, lhs, rhs)
}

pub fn sto(dst: OutputStream, src: Register) -> Instruction {
    Instruction::Store(dst, src)
}

#[derive(Debug, Clone, Copy, VariantCount)]
enum Opcode {
    Add,
    Load,
    Multiply,
    Store,
}

impl Opcode {
    // TODO: Test(s)
    pub fn from_u32(x: u32) -> Option<Opcode> {
        match x {
            0 => Some(Opcode::Add),
            1 => Some(Opcode::Load),
            2 => Some(Opcode::Multiply),
            3 => Some(Opcode::Store),
            _ => None
        }
    }
}

const_assert_eq!(Opcode::VARIANT_COUNT, Instruction::VARIANT_COUNT);

pub type EncodedInstruction = u32;

trait EncodedInstructionExtensions {
    fn clear(self, field: &Field) -> EncodedInstruction;
    fn set(self, field: &Field, value: EncodedInstruction) -> EncodedInstruction;
    fn value(self, field: &Field) -> EncodedInstruction;
}

impl EncodedInstructionExtensions for EncodedInstruction {
    fn clear(self, field: &Field) -> EncodedInstruction {
        self & !field.mask()
    }

    fn set(self, field: &Field, value: EncodedInstruction) -> EncodedInstruction {
        assert!(value.next_power_of_two() <= (1 << field.num_bits), "Value is too large for field");
        let cleared = self.clear(field);
        cleared | (value << field.offset_bits)
    }

    fn value(self, field: &Field) -> EncodedInstruction {
        (self & field.mask()) >> field.offset_bits
    }
}

#[derive(Clone)]
pub struct Field {
    offset_bits: usize,
    num_bits: usize,
}

impl Field {
    fn new(offset_bits: usize, num_bits: usize) -> Field {
        assert_ne!(num_bits, 0);
        assert!(num_bits < size_of::<EncodedInstruction>() * 8);

        Field {
            offset_bits,
            num_bits,
        }
    }

    fn mask(&self) -> EncodedInstruction {
        ((1 << self.num_bits) - 1) << self.offset_bits
    }
}

struct FieldFactory {
    offset_bits: usize,
}

impl FieldFactory {
    fn new() -> FieldFactory {
        FieldFactory {
            offset_bits: 0,
        }
    }

    fn field(&mut self, num_bits: usize) -> Field {
        let ret = Field::new(self.offset_bits, num_bits);
        self.offset_bits += num_bits;
        assert!(self.offset_bits <= size_of::<EncodedInstruction>() * 8, "Can't fit new field into encoding");
        ret
    }
}

pub struct Encoding {
    pub opcode: Field,
    pub dst_reg: Field,
    pub dst_output_stream: Field,
    pub src_reg: Field,
    pub src_input_stream: Field,
    pub lhs_reg: Field,
    pub rhs_reg: Field,
}

impl Encoding {
    pub fn new() -> Encoding {
        let mut f = FieldFactory::new();

        let num_bits = |variant_count: usize| {
            variant_count.next_power_of_two().trailing_zeros() as usize + 1
        };

        let opcode = f.field(num_bits(Instruction::VARIANT_COUNT));

        let reg_num_bits = num_bits(Register::VARIANT_COUNT);
        let dst_reg = f.field(reg_num_bits);
        let output_stream_num_bits = num_bits(OutputStream::VARIANT_COUNT);
        assert!(output_stream_num_bits <= reg_num_bits);
        let dst_output_stream = dst_reg.clone();
        let src_reg = f.field(reg_num_bits);
        let input_stream_num_bits = num_bits(InputStream::VARIANT_COUNT);
        assert!(input_stream_num_bits <= reg_num_bits);
        let src_input_stream = src_reg.clone();
        let lhs_reg = src_reg.clone();
        let rhs_reg = f.field(reg_num_bits);

        Encoding {
            opcode,
            dst_reg,
            dst_output_stream,
            src_reg,
            src_input_stream,
            lhs_reg,
            rhs_reg,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use super::InputStream::*;
    use super::Instruction::*;
    use super::OutputStream::*;
    use super::Register::*;

    #[test]
    fn assemble_and_roundtrip_add() {
        let i = add(R0, R1, R2);
        let e = Add(R0, R1, R2);
        assemble_and_roundtrip(i, e);
    }

    #[test]
    fn assemble_and_roundtrip_lod() {
        let i = lod(R3, I0);
        let e = Load(R3, I0);
        assemble_and_roundtrip(i, e);
    }

    #[test]
    fn assemble_and_roundtrip_mul() {
        let i = mul(R3, R4, R5);
        let e = Multiply(R3, R4, R5);
        assemble_and_roundtrip(i, e);
    }

    #[test]
    fn assemble_and_roundtrip_sto() {
        let i = sto(O0, R6);
        let e = Store(O0, R6);
        assemble_and_roundtrip(i, e);
    }

    fn assemble_and_roundtrip(i: Instruction, e: Instruction) {
        assert_eq!(i, e);
        let encoded = i.encode();
        let decoded = Instruction::decode(encoded).expect("Couldn't decode instruction");
        assert_eq!(decoded, e);
    }

    #[test]
    fn basic_fields() {
        let mut f = FieldFactory::new();

        let a = f.field(4);
        let b = f.field(4);
        let c = f.field(4);
        let d = f.field(4);

        assert_eq!(a.mask(), 0x000f);
        assert_eq!(b.mask(), 0x00f0);
        assert_eq!(c.mask(), 0x0f00);
        assert_eq!(d.mask(), 0xf000);

        assert_eq!(0xfade.clear(&a), 0xfad0);
        assert_eq!(0xfade.clear(&b), 0xfa0e);
        assert_eq!(0xfade.clear(&c), 0xf0de);
        assert_eq!(0xfade.clear(&d), 0x0ade);

        assert_eq!(0xfade.set(&a, 0xb), 0xfadb);
        assert_eq!(0xfade.set(&b, 0xb), 0xfabe);
        assert_eq!(0xfade.set(&c, 0xb), 0xfbde);
        assert_eq!(0xfade.set(&d, 0xb), 0xbade);

        assert_eq!(0xfade.value(&a), 0xe);
        assert_eq!(0xfade.value(&b), 0xd);
        assert_eq!(0xfade.value(&c), 0xa);
        assert_eq!(0xfade.value(&d), 0xf);
    }
}
