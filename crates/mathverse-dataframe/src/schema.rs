use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use crate::dtype::DType;
use crate::errors::{DataFrameError, DataFrameResult};

/// The schema of a DataFrame: ordered column names and their dtypes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    fields: Vec<Field>,
    /// O(log n) name → index lookup, rebuilt on mutation.
    name_idx: BTreeMap<String, usize>,
}

/// A single field in a schema: a column name and its dtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    name: String,
    dtype: DType,
}

impl Field {
    /// Creates a new field.
    #[must_use]
    pub fn new(name: impl Into<String>, dtype: DType) -> Self {
        Self {
            name: name.into(),
            dtype,
        }
    }

    /// Returns the field name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the field dtype.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }
}

impl Schema {
    /// Creates an empty schema.
    #[must_use]
    pub fn empty() -> Self {
        Self { fields: Vec::new(), name_idx: BTreeMap::new() }
    }

    /// Creates a schema from a list of (name, dtype) pairs.
    pub fn from_fields(fields: Vec<Field>) -> Self {
        let name_idx: BTreeMap<String, usize> = fields
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), i))
            .collect();
        Self { fields, name_idx }
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Returns `true` if the schema has no columns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Returns the field at `index`, or an error if out of bounds.
    #[must_use]
    pub fn field(&self, index: usize) -> DataFrameResult<&Field> {
        self.fields
            .get(index)
            .ok_or_else(|| DataFrameError::IndexOutOfBounds {
                index,
                length: self.fields.len(),
            })
    }

    /// Returns the index of a column by name — O(log n).
    #[must_use]
    pub fn index_of(&self, name: &str) -> DataFrameResult<usize> {
        self.name_idx
            .get(name)
            .copied()
            .ok_or_else(|| DataFrameError::ColumnNotFound(name.to_string()))
    }

    /// Returns the dtype of a column by name — O(log n).
    #[must_use]
    pub fn dtype_of(&self, name: &str) -> DataFrameResult<DType> {
        self.index_of(name).map(|i| self.fields[i].dtype)
    }

    /// Adds a field to the schema. Returns an error if the name already exists — O(log n).
    pub fn add_field(&mut self, field: Field) -> DataFrameResult<()> {
        if self.name_idx.contains_key(field.name.as_str()) {
            return Err(DataFrameError::DuplicateColumn(field.name));
        }
        let idx = self.fields.len();
        self.name_idx.insert(field.name.clone(), idx);
        self.fields.push(field);
        Ok(())
    }

    /// Removes a field by name, returning the removed field — O(n) shift.
    pub fn remove_field(&mut self, name: &str) -> DataFrameResult<Field> {
        let idx = self.index_of(name)?;
        self.name_idx.remove(name);
        let field = self.fields.remove(idx);
        // Rebuild index after shift
        self.name_idx.clear();
        for (i, f) in self.fields.iter().enumerate() {
            self.name_idx.insert(f.name.clone(), i);
        }
        Ok(field)
    }

    /// Renames a column — O(log n).
    pub fn rename(&mut self, old_name: &str, new_name: impl Into<String>) -> DataFrameResult<()> {
        let idx = self.index_of(old_name)?;
        self.name_idx.remove(old_name);
        self.fields[idx].name = new_name.into();
        self.name_idx.insert(self.fields[idx].name.clone(), idx);
        Ok(())
    }

    /// Returns an iterator over all field names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.fields.iter().map(|f| f.name.as_str())
    }

    /// Returns an iterator over all dtypes.
    pub fn dtypes(&self) -> impl Iterator<Item = DType> + '_ {
        self.fields.iter().map(|f| f.dtype)
    }

    /// Returns `true` if the schema contains a column with the given name — O(log n).
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.name_idx.contains_key(name)
    }

    /// Returns the underlying fields.
    #[must_use]
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Schema(")?;
        for (i, field) in self.fields.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", field.name, field.dtype)?;
        }
        write!(f, ")")
    }
}
