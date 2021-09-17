use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputStream {
    I0,
}

impl fmt::Display for InputStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputStream {
    O0,
}

impl fmt::Display for OutputStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Register {
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
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
            _ => None
        }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Instruction {
    Add(Register, Register, Register),
    Load(Register, InputStream),
    Move(Register, Register),
    Store(OutputStream, Register),
}

impl Instruction {
    // TODO: Test(s)
    pub fn mnemonic(&self) -> &'static str {
        match self {
            Instruction::Add(_, _, _) => "add",
            Instruction::Load(_, _) => "lod",
            Instruction::Move(_, _) => "mov",
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
            Instruction::Move(dst, src) => write!(f, "{} {}, {}", mnemonic, dst, src),
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

pub fn mov(dst: Register, src: Register) -> Instruction {
    Instruction::Move(dst, src)
}

pub fn str(dst: OutputStream, src: Register) -> Instruction {
    Instruction::Store(dst, src)
}

#[cfg(test)]
mod test {
    use super::*;
    use super::InputStream::*;
    use super::Instruction::*;
    use super::OutputStream::*;
    use super::Register::*;

    #[test]
    fn assemble_add() {
        assert_eq!(add(R0, R1, R2), Add(R0, R1, R2));
    }

    #[test]
    fn assemble_lod() {
        assert_eq!(lod(R3, I0), Load(R3, I0));
    }

    #[test]
    fn assemble_mov() {
        assert_eq!(mov(R3, R4), Move(R3, R4));
    }

    #[test]
    fn assemble_str() {
        assert_eq!(str(O0, R6), Store(O0, R6));
    }
}
