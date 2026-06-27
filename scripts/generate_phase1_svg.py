def generate_phase1_svg():
    svg = []
    svg.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 650" width="1200" height="650" style="background-color: #fafbfc; font-family: Arial, sans-serif;">')
    
    # Defs for arrowhead and gradients
    svg.append('<defs>')
    svg.append('<marker id="arrow_gray" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">')
    svg.append('<path d="M0,0 L0,6 L9,3 z" fill="#666" />')
    svg.append('</marker>')
    svg.append('<marker id="arrow_blue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">')
    svg.append('<path d="M0,0 L0,6 L9,3 z" fill="#1976D2" />')
    svg.append('</marker>')
    svg.append('<marker id="arrow_red" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">')
    svg.append('<path d="M0,0 L0,6 L9,3 z" fill="#D32F2F" />')
    svg.append('</marker>')
    
    svg.append('<linearGradient id="grad_xgboost" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#ffcc80" /><stop offset="100%" stop-color="#ff9800" /></linearGradient>')
    svg.append('<linearGradient id="grad_dag" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#a5d6a7" /><stop offset="100%" stop-color="#4caf50" /></linearGradient>')
    svg.append('<linearGradient id="grad_input" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#eeeeee" /><stop offset="100%" stop-color="#bdbdbd" /></linearGradient>')
    svg.append('<filter id="shadow" x="-10%" y="-10%" width="120%" height="120%"><feDropShadow dx="2" dy="4" stdDeviation="4" flood-opacity="0.15"/></filter>')
    svg.append('</defs>')

    # Helper function for boxes
    def draw_box(x, y, w, h, text_lines, grad_id, rx="10"):
        box = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" fill="url(#{grad_id})" filter="url(#shadow)" stroke="#444" stroke-width="2"/>'
        texts = ""
        for i, line in enumerate(text_lines):
            bold = 'font-weight="bold"' if i == 0 else ''
            font_size = "16" if i == 0 else "14"
            texts += f'<text x="{x + w/2}" y="{y + h/2 + (i*20) - ((len(text_lines)-1)*10)}" text-anchor="middle" dominant-baseline="middle" font-size="{font_size}" fill="#111" {bold}>{line}</text>'
        return box + texts

    def draw_line(x1, y1, x2, y2, color="#666", label="", marker="arrow_gray"):
        line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3" marker-end="url(#{marker})"/>'
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            line += f'<rect x="{mx-80}" y="{my-15}" width="160" height="30" fill="white" opacity="0.9" rx="5"/>'
            line += f'<text x="{mx}" y="{my}" text-anchor="middle" dominant-baseline="middle" font-size="12" font-weight="bold" fill="{color}">{label}</text>'
        return line
        
    def draw_curved_line(x1, y1, x2, y2, color="#666", marker="arrow_gray"):
        path = f'<path d="M {x1} {y1} C {x1+50} {y1}, {x2-50} {y2}, {x2} {y2}" fill="none" stroke="{color}" stroke-width="3" marker-end="url(#{marker})"/>'
        return path

    # Main Titles
    svg.append('<text x="600" y="40" text-anchor="middle" font-size="28" font-weight="bold" fill="#222">Phase 1 Evolution: Why We Switched to a PyTorch DAG</text>')
    
    # Dividers
    svg.append('<line x1="600" y1="80" x2="600" y2="600" stroke="#ccc" stroke-width="2" stroke-dasharray="8,4"/>')

    # ---------------- LEFT SIDE: XGBOOST ----------------
    svg.append('<text x="300" y="90" text-anchor="middle" font-size="22" font-weight="bold" fill="#e65100">THE PROBLEM: XGBoost Classifier Chain</text>')
    svg.append('<text x="300" y="115" text-anchor="middle" font-size="14" fill="#666">Individual decision trees suffering from Information Loss</text>')
    
    svg.append(draw_box(100, 160, 140, 60, ["Input Features", "(17-D Vector)"], "grad_input"))
    
    y_starts = [160, 260, 360, 460]
    labels = ["Diabetes Model", "CHF Model", "Sepsis Model", "AKI Model"]
    
    for i, (y, label) in enumerate(zip(y_starts, labels)):
        svg.append(draw_box(320, y, 160, 60, [label, "(Independent XGBoost)"], "grad_xgboost"))
        
        # Connect input to model
        svg.append(draw_curved_line(240, 190, 320, y + 30))
        
        # Output probability
        svg.append(f'<text x="510" y="{y + 30}" text-anchor="start" dominant-baseline="middle" font-size="14" font-weight="bold" fill="#d32f2f">► Output: 1D Probability (0 to 1)</text>')

    # Connect models (Classifier Chain)
    for i in range(3):
        x_center = 400
        y_top = y_starts[i] + 60
        y_bottom = y_starts[i+1]
        svg.append(draw_line(x_center, y_top, x_center, y_bottom, color="#D32F2F", label="1-D Binary Logit", marker="arrow_red"))

    # Summary box
    svg.append(draw_box(100, 560, 400, 60, ["FLAW: The Multi-Task Illusion", "Passing only a 1D probability loses all dense physiological context."], "grad_input"))


    # ---------------- RIGHT SIDE: PYTORCH DAG ----------------
    svg.append('<text x="900" y="90" text-anchor="middle" font-size="22" font-weight="bold" fill="#1b5e20">THE SOLUTION: PyTorch DAG Network</text>')
    svg.append('<text x="900" y="115" text-anchor="middle" font-size="14" fill="#666">A neural network passing dense contextual states downstream</text>')
    
    svg.append(draw_box(650, 160, 140, 60, ["Input Features", "(17-D Vector)"], "grad_input"))
    svg.append(draw_line(790, 190, 830, 190, color="#1976D2", marker="arrow_blue"))
    svg.append(draw_box(830, 160, 200, 60, ["Shared Global Trunk", "(64-Dimensional Latent State)"], "grad_input"))
    
    y_starts_dag = [260, 360, 460, 560]
    labels_dag = ["Diabetes Head", "CHF Head", "Sepsis Head", "AKI Head"]
    
    for i, (y, label) in enumerate(zip(y_starts_dag, labels_dag)):
        svg.append(draw_box(850, y, 160, 60, [label, "(32-D Hidden Layer)"], "grad_dag"))
        # Connection from trunk to heads
        svg.append(draw_curved_line(930, 220, 850, y + 30, color="#1976D2", marker="arrow_blue"))

    # Connect heads (Dense context)
    for i in range(3):
        x_center = 930
        y_top = y_starts_dag[i] + 60
        y_bottom = y_starts_dag[i+1]
        svg.append(draw_line(x_center, y_top, x_center, y_bottom, color="#1976D2", label="32-D Dense Context", marker="arrow_blue"))

    svg.append('</svg>')
    
    with open('../results/Evolution_Diagram.svg', 'w') as f:
        f.write("\n".join(svg))
        
generate_phase1_svg()
