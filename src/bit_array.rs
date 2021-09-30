use std::mem::size_of;

type StorageElement = u32;

const STORAGE_ELEMENT_BITS: usize = size_of::<StorageElement>();

pub struct BitArray {
    len: usize,
    storage: Vec<StorageElement>,
}

impl BitArray {
    pub fn new(len: usize) -> BitArray {
        BitArray {
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

    pub fn iter_set_indices(&self) -> BitArraySetIndexIterator<'_> {
        BitArraySetIndexIterator::new(self)
    }
}

#[derive(Clone)]
pub struct BitArraySetIndexIterator<'a> {
    bit_array: &'a BitArray,
    index: usize,
}

impl<'a> BitArraySetIndexIterator<'a> {
    fn new(bit_array: &'a BitArray) -> BitArraySetIndexIterator<'a> {
        BitArraySetIndexIterator {
            bit_array,
            index: 0,
        }
    }
}

impl<'a> Iterator for BitArraySetIndexIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.index >= self.bit_array.len {
                return None;
            }

            let index = self.index;
            self.index += 1;

            let (storage_index, mask) = storage_index_and_mask(index);
            if (self.bit_array.storage[storage_index] & mask) != 0 {
                return Some(index);
            }
        }
    }
}

fn storage_index_and_mask(index: usize) -> (usize, StorageElement) {
    let storage_index = index / STORAGE_ELEMENT_BITS;
    let bit_index = index % STORAGE_ELEMENT_BITS;
    let mask = 1 << bit_index;
    (storage_index, mask)
}
