//! `peregrine-tools`: the built-in tools (`read`, `write`, `edit`, `bash`)
//! and the host `Ops` implementation they run on.
//!
//! Tools depend on `Ops`, never on the filesystem or process spawning
//! directly — `SandboxOps` plugs into the same trait in phase 2.

pub mod local_ops;
pub mod tools;

pub use local_ops::{LocalOps, MAX_FILE_BYTES};
pub use tools::{BashTool, EditTool, ReadTool, WriteTool, builtin_tools, register_builtin};
