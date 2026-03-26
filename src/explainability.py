

def tree_visualizer(model, feature_names=None):
    '''
    La función genera una representación visual del árbol aprendido para que sea fácil de interpretar para el usuario
    '''
    if model.root is None:
        return "El árbol aun no fue entrenado"
    
    dot = "digraph Tree {\nnode [shape=box, style=\"filled, rounded\", color=\"black\", fontname=helvetica] ;\n"

    def traverse(node, node_id):
        nonlocal dot
        
        # Información actual
        gini_str = f"Gini: {node.gini:.3f}" if node.gini is not None else ""
        pacientes_str = f"Samples: {node.pacientes_totales}" if node.pacientes_totales is not None else ""

        if node.is_leaf_node():
            etiqueta = f"HOJA\\n{gini_str}\\n{pacientes_str}\\nPredicción: {node.value}"
            dot += f'{node_id} [label="{etiqueta}", fillcolor="#e5f5e0"] ;\n'
            return node_id + 1
        
        # Si no es un nodo hoja, significa que hubo una división
        # Nos quedamos con la información de la división
        caracteristica = f"X[{node.feature}]" if feature_names is None else feature_names[node.feature]
        etiqueta = f"{caracteristica} <= {node.threshold:.2f}\\n{gini_str}\\n{pacientes_str}"
        dot += f'{node_id} [label="{etiqueta}", fillcolor="#3498db66"] ;\n'
        
        # Actualizo para seguir con el dfs
        parent_id = node_id
        child_id = node_id + 1

        dot += f'{parent_id} -> {child_id} [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n'
        child_id = traverse(node.left, child_id) # Vuelvo a ejecutar la función siguiendo el dfs con el próximo del árbol

        dot += f'{parent_id} -> {child_id} [labeldistance=2.5, labelangle=-45, headlabel="False"] ;\n'
        child_id = traverse(node.right, child_id)

        return child_id

    traverse(model.root, 0)
    dot += "}"
    return dot