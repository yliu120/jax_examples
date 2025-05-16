import jax


def replace_pytree_leaves(
    old_tree: jax.tree_util.PyTreeDef, new_tree: jax.tree_util.PyTreeDef
):
    """Replaces the leaves of the old tree with the new tree."""
    # We cannot assert that the tree structures are the same here because there are
    # some unique runtime objects probably included in the tree structure.
    old_leaves = jax.tree.leaves(old_tree)
    new_leaves = jax.tree.leaves(new_tree)
    assert len(old_leaves) == len(new_leaves)
    _, treedef = jax.tree.flatten(old_tree)
    return jax.tree.unflatten(treedef, new_leaves)
