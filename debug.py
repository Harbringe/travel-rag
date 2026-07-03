from chunking import build_documents

parents, children = build_documents()

with open("debug_chunks.md", "w", encoding="utf-8") as f:

    f.write("# RAG Chunk Debug Output\n\n")

    f.write(f"- Parent Sections: {len(parents)}\n")
    f.write(f"- Child Chunks: {len(children)}\n\n")

    # ==================================================
    # PARENTS
    # ==================================================
    f.write("# Parent Sections\n\n")

    for pid, parent in parents.items():

        f.write(f"## {pid}\n\n")

        for k, v in parent.metadata.items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n### Content\n\n")
        f.write("```text\n")
        f.write(parent.page_content)
        f.write("\n```\n\n")

    # ==================================================
    # CHILDREN
    # ==================================================
    f.write("\n# Child Chunks\n\n")

    for idx, child in enumerate(children):

        f.write(f"## Chunk {idx}\n\n")

        for k, v in child.metadata.items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n### Chunk Content\n\n")
        f.write("```text\n")
        f.write(child.page_content)
        f.write("\n```\n\n---\n\n")

print("Written to debug_chunks.md")