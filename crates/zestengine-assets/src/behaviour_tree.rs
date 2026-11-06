//! Behaviour-tree asset (`.behaviour`): a RON document wrapping a
//! `zestengine_core::ai::BtNode` graph. The runtime evaluator and the
//! `Behaviour Tree` component live in `zestengine-core`; this module is the
//! on-disk asset format plus its load/save helpers.

use std::path::Path;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

use zestengine_core::ai::BtNode;

pub const BEHAVIOUR_TREE_EXTENSION: &str = "behaviour";

/// A behaviour-tree document. `name` is cosmetic; `root` is the node graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BehaviourTree {
    pub name: String,
    pub root: BtNode,
}

impl Default for BehaviourTree {
    fn default() -> Self {
        Self {
            name: "New Behaviour".into(),
            root: BtNode::default(),
        }
    }
}

pub fn load_behaviour_tree(path: impl AsRef<Path>) -> Result<BehaviourTree> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("parsing behaviour tree {}", path.display()))
}

pub fn save_behaviour_tree(path: impl AsRef<Path>, tree: &BehaviourTree) -> Result<()> {
    let path = path.as_ref();
    let text = ron::ser::to_string_pretty(tree, ron::ser::PrettyConfig::default())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_core::ai::{BtAction, BtNode};

    #[test]
    fn round_trips_through_ron() {
        let tree = BehaviourTree {
            root: BtNode::Sequence {
                children: vec![
                    BtNode::Condition {
                        key: "has_target".into(),
                        value: true,
                        invert: false,
                    },
                    BtNode::Action {
                        action: BtAction::Chase,
                        param: "target_id".into(),
                    },
                ],
            },
            ..Default::default()
        };
        let text = ron::ser::to_string_pretty(&tree, ron::ser::PrettyConfig::default()).unwrap();
        let back: BehaviourTree = ron::from_str(&text).unwrap();
        assert_eq!(back, tree);
    }
}
