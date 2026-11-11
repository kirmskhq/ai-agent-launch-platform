from agent_platform.loader import load_graph


def test_relative_policy_import_follows_the_code_version(workspace):
    folder = workspace / "reload-example"
    folder.mkdir()
    policy = folder / "policy.py"
    policy.write_text("VALUE = 1\n")
    (folder / "graph.py").write_text(
        "from .policy import VALUE\n"
        "class Graph:\n"
        "    def invoke(self, state):\n"
        "        return {'output': {'value': VALUE}}\n"
        "def build_graph():\n"
        "    return Graph()\n"
    )
    assert load_graph(folder).invoke({})["output"]["value"] == 1
    policy.write_text("VALUE = 22\n")
    assert load_graph(folder).invoke({})["output"]["value"] == 22
