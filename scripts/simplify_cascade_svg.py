def generate_cascade_svg():
    svg = []
    svg.append('<?xml version="1.0" encoding="utf-8"?>')
    svg.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 850" width="1000" height="850" style="background-color: #ffffff; font-family: Arial, sans-serif;">')
    
    svg.append('<defs>')
    # Arrow heads
    svg.append('<marker id="arrow_blue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">')
    svg.append('<path d="M0,0 L0,6 L9,3 z" fill="#1E88E5" />')
    svg.append('</marker>')
    # Gradients
    svg.append('<linearGradient id="grad_chronic" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#E8F5E9" /><stop offset="100%" stop-color="#C8E6C9" /></linearGradient>')
    svg.append('<linearGradient id="grad_acute" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#FFEBEE" /><stop offset="100%" stop-color="#FFCDD2" /></linearGradient>')
    svg.append('<linearGradient id="grad_link" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#FFFDE7" /><stop offset="100%" stop-color="#FFF9C4" /></linearGradient>')
    svg.append('</defs>')

    # Title
    svg.append('<text x="500" y="40" text-anchor="middle" font-size="26" font-weight="bold" fill="#1A237E">Clinical Comorbidity Cascade (Simple Explanation)</text>')
    svg.append('<text x="500" y="65" text-anchor="middle" font-size="14" fill="#555">How chronic conditions sequentially trigger acute physiological failure</text>')

    # Draw a Disease Block helper
    def draw_disease_node(x, y, title, subtype, bullets, color_grad):
        svg_chunk = []
        svg_chunk.append(f'<rect x="{x}" y="{y}" width="340" height="130" rx="12" ry="12" fill="url(#{color_grad})" stroke="#37474F" stroke-width="2"/>')
        svg_chunk.append(f'<text x="{x+20}" y="{y+30}" font-size="12" font-weight="bold" fill="#555" letter-spacing="1">{subtype.upper()}</text>')
        svg_chunk.append(f'<text x="{x+20}" y="{y+55}" font-size="20" font-weight="bold" fill="#212121">{title}</text>')
        for idx, bullet in enumerate(bullets):
            svg_chunk.append(f'<text x="{x+20}" y="{y+85 + (idx*20)}" font-size="13" fill="#37474F">- {bullet}</text>')
        return "\n".join(svg_chunk)

    # Draw a link explanation helper (Modified to adjust height dynamically for simple text)
    def draw_link_box(x, y, header, lines):
        svg_chunk = []
        svg_chunk.append(f'<rect x="{x}" y="{y}" width="360" height="105" rx="8" ry="8" fill="url(#grad_link)" stroke="#FBC02D" stroke-width="1.5"/>')
        svg_chunk.append(f'<text x="{x+15}" y="{y+25}" font-size="13" font-weight="bold" fill="#F57F17">{header.upper()}</text>')
        for idx, line in enumerate(lines):
            svg_chunk.append(f'<text x="{x+15}" y="{y+48 + (idx*18)}" font-size="12.5" fill="#3E2723">{line}</text>')
        return "\n".join(svg_chunk)

    # Coordinates and content
    # NODES
    svg.append(draw_disease_node(50, 100, "1. Diabetes", "Chronic Metabolic", 
                                 ["High blood sugar damages blood vessels.", "Makes arteries stiff and narrow over time."], "grad_chronic"))
    
    svg.append(draw_disease_node(50, 290, "2. Congestive Heart Failure", "Chronic Cardiovascular", 
                                 ["The heart becomes too weak to pump properly.", "Causes fluid to pool inside the lungs and body."], "grad_chronic"))
    
    svg.append(draw_disease_node(50, 480, "3. Sepsis", "Acute Systemic Infection", 
                                 ["A severe, body-wide response to infection.", "Causes blood pressure to drop dangerously low."], "grad_acute"))
    
    svg.append(draw_disease_node(50, 670, "4. Acute Kidney Injury", "Acute Secondary Failure", 
                                 ["Kidneys suddenly stop filtering waste.", "Spikes toxic waste markers (Creatinine)."], "grad_acute"))

    # CONNECTIONS & LINK EXPLANATIONS
    # Link 1: Diabetes -> CHF
    svg.append('<line x1="220" y1="230" x2="220" y2="280" stroke="#1E88E5" stroke-width="3" marker-end="url(#arrow_blue)"/>')
    svg.append(draw_link_box(450, 115, "Connection 1: Diabetes to CHF", 
                             ["High blood sugar damages and stiffens blood vessels.", "The heart must pump much harder to push blood through,", "which eventually weakens the heart muscle (CHF)."]))

    # Link 2: CHF -> Sepsis
    svg.append('<line x1="220" y1="420" x2="220" y2="470" stroke="#1E88E5" stroke-width="3" marker-end="url(#arrow_blue)"/>')
    svg.append(draw_link_box(450, 305, "Connection 2: CHF to Sepsis", 
                             ["A weak heart causes fluid to pool in the lungs.", "This trapped fluid is a breeding ground for bacteria,", "making it easy for standard lung infections to spread body-wide."]))

    # Link 3: Sepsis -> AKI
    svg.append('<line x1="220" y1="610" x2="220" y2="660" stroke="#1E88E5" stroke-width="3" marker-end="url(#arrow_blue)"/>')
    svg.append(draw_link_box(450, 495, "Connection 3: Sepsis to AKI", 
                             ["Sepsis causes a sudden, severe drop in blood pressure.", "Without enough pressure, blood flow to the kidneys collapses,", "starving kidney cells of oxygen and causing failure."]))

    # Final visual link connecting explanations together
    svg.append('<path d="M 410 165 L 410 545" fill="none" stroke="#B0BEC5" stroke-width="2" stroke-dasharray="5,5"/>')

    svg.append('</svg>')
    
    with open('../results/Disease_Cascade_Diagram.svg', 'w') as f:
        f.write("\n".join(svg))

generate_cascade_svg()
print("SUCCESS")
