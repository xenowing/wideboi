const STORAGE_ELEMENT_BITS: usize = 32;

pub struct BitVector {
    len: usize,
    storage: Vec<u32>,
}

impl BitVector {
    pub fn new(len: usize) -> BitVector {
        BitVector {
            len,
            storage: vec![0; (len + (STORAGE_ELEMENT_BITS - 1)) / STORAGE_ELEMENT_BITS],
        }
    }

    pub fn clear(&mut self, index: usize) {
        let (storage_index, mask) = storage_index_and_mask(index);
        self.storage[storage_index] &= !mask;
    }

    pub fn set(&mut self, index: usize) {
        let (storage_index, mask) = storage_index_and_mask(index);
        self.storage[storage_index] |= mask;
    }

    pub fn iter_set_indices(&self) -> BitVectorSetIndexIterator<'_> {
        BitVectorSetIndexIterator::new(self)
    }
}

#[derive(Clone)]
pub struct BitVectorSetIndexIterator<'a> {
    bit_vector: &'a BitVector,
    index: usize,
}

impl<'a> BitVectorSetIndexIterator<'a> {
    fn new(bit_vector: &'a BitVector) -> BitVectorSetIndexIterator<'a> {
        BitVectorSetIndexIterator {
            bit_vector,
            index: 0,
        }
    }
}

impl<'a> Iterator for BitVectorSetIndexIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.index >= self.bit_vector.len {
                return None;
            }

            let index = self.index;
            self.index += 1;

            let (storage_index, mask) = storage_index_and_mask(index);
            if (self.bit_vector.storage[storage_index] & mask) != 0 {
                return Some(index);
            }
        }
    }
}

fn storage_index_and_mask(index: usize) -> (usize, u32) {
    let storage_index = index / STORAGE_ELEMENT_BITS;
    let bit_index = index % STORAGE_ELEMENT_BITS;
    let mask = 1 << bit_index;
    (storage_index, mask)
}
