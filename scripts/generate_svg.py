def generate_svg():
    svg = []
    svg.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600" width="1000" height="600" style="background-color: white; font-family: Arial, sans-serif;">')
    
    # Defs for arrowhead and gradients
    svg.append('<defs>')
    svg.append('<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">')
    svg.append('<path d="M0,0 L0,6 L9,3 z" fill="#333" />')
    svg.append('</marker>')
    svg.append('<marker id="arrow_red" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">')
    svg.append('<path d="M0,0 L0,6 L9,3 z" fill="#E53935" />')
    svg.append('</marker>')
    
    svg.append('<linearGradient id="grad_input" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#f5f7fa" /><stop offset="100%" stop-color="#c3cfe2" /></linearGradient>')
    svg.append('<linearGradient id="grad_trunk" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#e0c3fc" /><stop offset="100%" stop-color="#8ec5fc" /></linearGradient>')
    svg.append('<linearGradient id="grad_head" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#d4fc79" /><stop offset="100%" stop-color="#96e6a1" /></linearGradient>')
    svg.append('<linearGradient id="grad_out" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#fdfbfb" /><stop offset="100%" stop-color="#ebedee" /></linearGradient>')
    svg.append('<filter id="shadow" x="-10%" y="-10%" width="120%" height="120%"><feDropShadow dx="2" dy="4" stdDeviation="4" flood-opacity="0.15"/></filter>')
    svg.append('</defs>')

    # Helper function for boxes
    def draw_box(x, y, w, h, text_lines, grad_id):
        box = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15" ry="15" fill="url(#{grad_id})" filter="url(#shadow)" stroke="#444" stroke-width="2"/>'
        texts = ""
        for i, line in enumerate(text_lines):
            bold = 'font-weight="bold"' if i == 0 else ''
            font_size = "16" if i == 0 else "14"
            texts += f'<text x="{x + w/2}" y="{y + h/2 + (i*20) - ((len(text_lines)-1)*10)}" text-anchor="middle" dominant-baseline="middle" font-size="{font_size}" fill="#222" {bold}>{line}</text>'
        return box + texts

    def draw_ellipse(cx, cy, rx, ry, text_lines, grad_id):
        el = f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="url(#{grad_id})" filter="url(#shadow)" stroke="#444" stroke-width="2"/>'
        texts = ""
        for i, line in enumerate(text_lines):
            bold = 'font-weight="bold"' if i == 0 else ''
            font_size = "14" if i == 0 else "12"
            texts += f'<text x="{cx}" y="{cy + (i*20) - ((len(text_lines)-1)*10)}" text-anchor="middle" dominant-baseline="middle" font-size="{font_size}" fill="#222" {bold}>{line}</text>'
        return el + texts

    def draw_line(x1, y1, x2, y2, color="#333", dash=False, label=""):
        style = 'stroke-dasharray="8,4"' if dash else ''
        marker = 'url(#arrow_red)' if color == '#E53935' else 'url(#arrow)'
        line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3" {style} marker-end="{marker}"/>'
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            line += f'<rect x="{mx+5}" y="{my-15}" width="110" height="30" fill="white" opacity="0.8" rx="5"/>'
            line += f'<text x="{mx+60}" y="{my}" text-anchor="middle" dominant-baseline="middle" font-size="12" font-weight="bold" fill="{color}">{label}</text>'
        return line

    def draw_curved_line(x1, y1, x2, y2):
        path = f'<path d="M {x1} {y1} C {x1+100} {y1}, {x2-100} {y2}, {x2} {y2}" fill="none" stroke="#333" stroke-width="3" marker-end="url(#arrow)"/>'
        return path

    # Title
    svg.append('<text x="500" y="40" text-anchor="middle" font-size="24" font-weight="bold" fill="#111">Multi-Task Learning DAG Architecture</text>')

    # Components
    svg.append(draw_box(40, 240, 160, 80, ["Input Features", "(17-D Vector)"], "grad_input"))
    svg.append(draw_box(280, 240, 180, 80, ["Shared Global Trunk", "(64-D Latent State)"], "grad_trunk"))

    # Connections
    svg.append(draw_line(200, 280, 280, 280))

    # Disease Heads
    y_starts = [100, 200, 300, 400]
    labels = ["Diabetes", "CHF", "Sepsis", "AKI"]
    for i, (y, label) in enumerate(zip(y_starts, labels)):
        svg.append(draw_box(560, y, 160, 60, [f"{label} Head", "(32-D Hidden)"], "grad_head"))
        svg.append(draw_ellipse(880, y + 30, 80, 30, [f"{label}", "Prediction"], "grad_out"))
        
        # Trunk to Head (Curved)
        svg.append(draw_curved_line(460, 280, 560, y + 30))
        
        # Head to Output
        svg.append(draw_line(720, y + 30, 800, y + 30))

    # Stop Gradient Connections
    for i in range(3):
        x_center = 640
        y_top = y_starts[i] + 60
        y_bottom = y_starts[i+1]
        svg.append(draw_line(x_center, y_top, x_center, y_bottom, color="#E53935", dash=True, label="Stop-Gradient"))

    svg.append('</svg>')
    
    with open('../results/Architecture_Diagram.svg', 'w') as f:
        f.write("\n".join(svg))
        
generate_svg()
