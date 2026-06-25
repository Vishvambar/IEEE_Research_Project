import graphviz
import sys

try:
    dot = graphviz.Digraph(comment='DAG MTL Architecture', format='png')
    dot.attr(rankdir='TB', nodesep='0.8', ranksep='1.0', dpi='300')
    
    # Global attributes
    dot.attr('node', fontname='Arial', fontsize='12')
    dot.attr('edge', fontname='Arial', fontsize='10')

    # Nodes
    dot.node('Input', 'Input Features\n(17-D)', shape='box', style='filled', fillcolor='#EAEAEA', width='2')
    dot.node('Trunk', 'Shared Global Trunk\n(64-D)', shape='box', style='filled', fillcolor='#D0E4F5', width='2')

    # Disease Heads
    dot.node('Diab', 'Diabetes Head\n(32-D)', shape='box', style='filled', fillcolor='#E2F0D9')
    dot.node('CHF', 'CHF Head\n(32-D)', shape='box', style='filled', fillcolor='#E2F0D9')
    dot.node('Sepsis', 'Sepsis Head\n(32-D)', shape='box', style='filled', fillcolor='#E2F0D9')
    dot.node('AKI', 'AKI Head\n(32-D)', shape='box', style='filled', fillcolor='#E2F0D9')

    # Outputs
    dot.node('OutDiab', 'Diabetes\nLogit', shape='ellipse', fillcolor='white', style='filled')
    dot.node('OutCHF', 'CHF\nLogit', shape='ellipse', fillcolor='white', style='filled')
    dot.node('OutSepsis', 'Sepsis\nLogit', shape='ellipse', fillcolor='white', style='filled')
    dot.node('OutAKI', 'AKI\nLogit', shape='ellipse', fillcolor='white', style='filled')

    # Connections
    dot.edge('Input', 'Trunk')
    
    dot.edge('Trunk', 'Diab')
    dot.edge('Trunk', 'CHF')
    dot.edge('Trunk', 'Sepsis')
    dot.edge('Trunk', 'AKI')

    dot.edge('Diab', 'OutDiab')
    dot.edge('CHF', 'OutCHF')
    dot.edge('Sepsis', 'OutSepsis')
    dot.edge('AKI', 'OutAKI')

    # Stop gradients
    dot.edge('Diab', 'CHF', label=' Stop-Gradient\n(tensor.detach)', style='dashed', color='#FF6B6B', fontcolor='#D32F2F', constraint='false')
    dot.edge('CHF', 'Sepsis', label=' Stop-Gradient\n(tensor.detach)', style='dashed', color='#FF6B6B', fontcolor='#D32F2F', constraint='false')
    dot.edge('Sepsis', 'AKI', label=' Stop-Gradient\n(tensor.detach)', style='dashed', color='#FF6B6B', fontcolor='#D32F2F', constraint='false')

    # Force alignment
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Diab')
        s.node('CHF')
        s.node('Sepsis')
        s.node('AKI')

    dot.render('../results/Architecture_Diagram', view=False)
    print("SUCCESS")
except Exception as e:
    print("ERROR:", str(e))
    sys.exit(1)
