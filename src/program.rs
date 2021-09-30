use crate::instructions::*;

use std::collections::HashMap;

pub struct Program {
    pub statements: Vec<Statement>,
    pub num_variables: u32,
    pub num_input_stream_loads: HashMap<InputStream, u32>,
    pub num_output_stream_stores: u32,
}

impl Program {
    pub fn new() -> Program {
        Program {
            statements: Vec::new(),
            num_variables: 0,
            num_input_stream_loads: HashMap::new(),
            num_output_stream_stores: 0,
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
        // TODO: Limit loads based on offset field bits
        let loads = self.num_input_stream_loads.entry(src).or_insert(0);
        let offset = *loads;
        *loads += 1;
        self.statements.push(Statement::Load(name, src, offset as _));
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
        // TODO: Limit stores based on offset field bits
        let offset = self.num_output_stream_stores;
        self.num_output_stream_stores += 1;
        self.statements.push(Statement::Store(dst, src, offset as _));
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
    Load(VariableIndex, InputStream, u8),
    Multiply(VariableIndex, Scalar, Scalar, u8),
    Store(OutputStream, Scalar, u8),
}

#[derive(Clone)]
pub struct V3 {
    pub x: Scalar,
    pub y: Scalar,
    pub z: Scalar,
}

#[derive(Clone, Copy)]
pub struct VariableIndex(pub u32);
