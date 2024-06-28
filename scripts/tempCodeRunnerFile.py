    # from RLib.utils.serializers import serialize_table
    # import json

    # for agent in agents:
    #     q_table = agent.q_table
    #     path = agent.best_path()
    #     q_table_for_sp = get_q_table_for_path(q_table, path)
    #     serialized_q_table_for_sp = serialize_table(q_table_for_sp)
    #     json_q_table_for_sp = json.dumps(serialized_q_table_for_sp, indent=4)
    #     with open(os.path.join(RESULTS_DIR, f"q_star_for_shortest_path_{city_name}_{orig_node}-{dest_node}_{agent.strategy}.json"), "w") as f:
    #         f.write(json_q_table_for_sp)
    #         f.close()
