//! Pattern trie for N-gram analysis of instruction sequences.

use std::collections::HashMap;

/// A node in the pattern trie.
pub struct TrieNode {
    pub op: String,
    pub count: u64,
    pub depth: usize,
    pub children: HashMap<String, TrieNode>,
}

impl TrieNode {
    fn new(op: &str, depth: usize) -> Self {
        Self {
            op: String::from(op),
            count: 0,
            depth,
            children: HashMap::new(),
        }
    }
}

/// Pattern trie built from N-gram statistics.
pub struct PatternTrie {
    pub root: TrieNode,
    pub total_instructions: u64,
    pub window_size: usize,
}

/// A candidate pattern extracted from the trie.
pub struct PatternCandidate {
    pub pattern: Vec<String>,
    pub count: u64,
}

impl PatternTrie {
    pub fn new(total_instructions: u64, window_size: usize) -> Self {
        Self {
            root: TrieNode::new("", 0),
            total_instructions,
            window_size,
        }
    }

    pub fn insert(&mut self, ops: &[&str], count: u64) {
        let mut node = &mut self.root;
        for (i, op) in ops.iter().enumerate() {
            let depth = i + 1;
            node = node
                .children
                .entry(String::from(*op))
                .or_insert_with(|| TrieNode::new(op, depth));
        }
        node.count += count;
    }

    pub fn print_tree(&self, max_depth: usize, min_count: u64) {
        eprintln!(
            "Pattern Trie (total instructions: {})",
            self.total_instructions
        );
        eprintln!("{}", "-".repeat(70));
        self.print_node(&self.root, max_depth, min_count);
    }

    fn print_node(&self, node: &TrieNode, max_depth: usize, min_count: u64) {
        let mut children: Vec<_> = node.children.values().collect();
        children.sort_by(|a, b| b.count.cmp(&a.count));

        for child in children {
            if child.count < min_count || child.depth > max_depth {
                continue;
            }

            let pct = if self.total_instructions > 0 {
                (child.count as f64 / self.total_instructions as f64) * 100.0
            } else {
                0.0
            };

            let indent = "  ".repeat(child.depth - 1);
            let has_children = !child.children.is_empty()
                && child.depth < max_depth
                && child.children.values().any(|c| c.count >= min_count);
            let marker = if has_children { "+" } else { "-" };

            eprintln!(
                "{}{} {} (count: {}, {:.2}%)",
                indent, marker, child.op, child.count, pct
            );

            if child.depth < max_depth {
                self.print_node(child, max_depth, min_count);
            }
        }
    }

    pub fn collect_candidates(&self, min_depth: usize, min_count: u64) -> Vec<PatternCandidate> {
        let mut candidates = Vec::new();
        self.walk_candidates(
            &self.root,
            &mut Vec::new(),
            min_depth,
            min_count,
            &mut candidates,
        );
        candidates
    }

    fn walk_candidates(
        &self,
        node: &TrieNode,
        path: &mut Vec<String>,
        min_depth: usize,
        min_count: u64,
        out: &mut Vec<PatternCandidate>,
    ) {
        for child in node.children.values() {
            path.push(child.op.clone());

            if child.depth >= min_depth && child.count >= min_count {
                out.push(PatternCandidate {
                    pattern: path.clone(),
                    count: child.count,
                });
            }

            self.walk_candidates(child, path, min_depth, min_count, out);
            path.pop();
        }
    }

    pub fn depth_stats(&self) -> HashMap<usize, usize> {
        let mut stats: HashMap<usize, usize> = HashMap::new();
        self.count_depths(&self.root, &mut stats);
        stats
    }

    fn count_depths(&self, node: &TrieNode, stats: &mut HashMap<usize, usize>) {
        for child in node.children.values() {
            if child.count > 0 {
                *stats.entry(child.depth).or_insert(0) += 1;
            }
            self.count_depths(child, stats);
        }
    }
}
