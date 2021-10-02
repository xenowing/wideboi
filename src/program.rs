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

    pub fn dot_v3(&mut self, lhs: V3, rhs: V3, shift: u8) -> Scalar {
        let temp = self.mul_v3(lhs, rhs, shift);
        let lhs = self.add_s(temp.x, temp.y);
        self.add_s(lhs, temp.z)
    }

    pub fn dot_v4(&mut self, lhs: V4, rhs: V4, shift: u8) -> Scalar {
        let temp = self.mul_v4(lhs, rhs, shift);
        let lhs = self.add_s(temp.x, temp.y);
        let lhs = self.add_s(lhs, temp.z);
        self.add_s(lhs, temp.w)
    }

    pub fn load_s(&mut self, src: InputStream) -> Scalar {
        let t = self.alloc_var();
        // TODO: Limit loads based on offset field bits
        let loads = self.num_input_stream_loads.entry(src).or_insert(0);
        let offset = *loads;
        *loads += 1;
        self.statements.push(Statement::Load(t, src, offset as _));
        Scalar::VariableRef(t)
    }

    pub fn load_v3(&mut self, src: InputStream) -> V3 {
        V3 {
            x: self.load_s(src),
            y: self.load_s(src),
            z: self.load_s(src),
        }
    }

    pub fn load_v4(&mut self, src: InputStream) -> V4 {
        V4 {
            x: self.load_s(src),
            y: self.load_s(src),
            z: self.load_s(src),
            w: self.load_s(src),
        }
    }

    pub fn load_m3(&mut self, src: InputStream) -> M3 {
        M3 {
            columns: [
                self.load_v3(src),
                self.load_v3(src),
                self.load_v3(src),
            ],
        }
    }

    pub fn load_m4(&mut self, src: InputStream) -> M4 {
        M4 {
            columns: [
                self.load_v4(src),
                self.load_v4(src),
                self.load_v4(src),
                self.load_v4(src),
            ],
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

    pub fn mul_v4(&mut self, lhs: V4, rhs: V4, shift: u8) -> V4 {
        V4 {
            x: self.mul_s(lhs.x, rhs.x, shift),
            y: self.mul_s(lhs.y, rhs.y, shift),
            z: self.mul_s(lhs.z, rhs.z, shift),
            w: self.mul_s(lhs.w, rhs.w, shift),
        }
    }

    pub fn mul_m3_v3(&mut self, lhs: M3, rhs: V3, shift: u8) -> V3 {
        let rows = lhs.rows();
        V3 {
            x: self.dot_v3(rows[0], rhs, shift),
            y: self.dot_v3(rows[1], rhs, shift),
            z: self.dot_v3(rows[2], rhs, shift),
        }
    }

    pub fn mul_m4_v4(&mut self, lhs: M4, rhs: V4, shift: u8) -> V4 {
        let rows = lhs.rows();
        V4 {
            x: self.dot_v4(rows[0], rhs, shift),
            y: self.dot_v4(rows[1], rhs, shift),
            z: self.dot_v4(rows[2], rhs, shift),
            w: self.dot_v4(rows[3], rhs, shift),
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

    pub fn store_v4(&mut self, dst: OutputStream, src: V4) {
        self.store_s(dst, src.x);
        self.store_s(dst, src.y);
        self.store_s(dst, src.z);
        self.store_s(dst, src.w);
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Scalar {
    VariableRef(VariableIndex),
}

#[derive(Debug)]
pub enum Statement {
    Add(VariableIndex, Scalar, Scalar),
    Load(VariableIndex, InputStream, u8),
    Multiply(VariableIndex, Scalar, Scalar, u8),
    Store(OutputStream, Scalar, u8),
}

#[derive(Clone, Copy)]
pub struct V3 {
    pub x: Scalar,
    pub y: Scalar,
    pub z: Scalar,
}

#[derive(Clone, Copy)]
pub struct M3 {
    pub columns: [V3; 3],
}

impl M3 {
    pub fn rows(self) -> [V3; 3] {
        [
            V3 { x: self.columns[0].x, y: self.columns[1].x, z: self.columns[2].x },
            V3 { x: self.columns[0].y, y: self.columns[1].y, z: self.columns[2].y },
            V3 { x: self.columns[0].z, y: self.columns[1].z, z: self.columns[2].z },
        ]
    }

    pub fn transpose(self) -> M3 {
        M3 {
            columns: self.rows(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct V4 {
    pub x: Scalar,
    pub y: Scalar,
    pub z: Scalar,
    pub w: Scalar,
}

#[derive(Clone, Copy)]
pub struct M4 {
    pub columns: [V4; 4],
}

impl M4 {
    pub fn rows(self) -> [V4; 4] {
        [
            V4 { x: self.columns[0].x, y: self.columns[1].x, z: self.columns[2].x, w: self.columns[3].x },
            V4 { x: self.columns[0].y, y: self.columns[1].y, z: self.columns[2].y, w: self.columns[3].y },
            V4 { x: self.columns[0].z, y: self.columns[1].z, z: self.columns[2].z, w: self.columns[3].z },
            V4 { x: self.columns[0].w, y: self.columns[1].w, z: self.columns[2].w, w: self.columns[3].w },
        ]
    }

    pub fn transpose(self) -> M4 {
        M4 {
            columns: self.rows(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VariableIndex(pub u32);
