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
    svg.append('<text x="500" y="40" text-anchor="middle" font-size="26" font-weight="bold" fill="#1A237E">Clinical Comorbidity Cascade (Biological Dependency Flow)</text>')
    svg.append('<text x="500" y="65" text-anchor="middle" font-size="14" fill="#555">How chronic conditions sequentially trigger acute physiological failure</text>')

    # Draw a Disease Block helper (Removed filter for maximum compatibility)
    def draw_disease_node(x, y, title, subtype, bullets, color_grad):
        svg_chunk = []
        # Main Node Box
        svg_chunk.append(f'<rect x="{x}" y="{y}" width="340" height="130" rx="12" ry="12" fill="url(#{color_grad})" stroke="#37474F" stroke-width="2"/>')
        # Subtype Tag
        svg_chunk.append(f'<text x="{x+20}" y="{y+30}" font-size="12" font-weight="bold" fill="#555" letter-spacing="1">{subtype.upper()}</text>')
        # Disease Title
        svg_chunk.append(f'<text x="{x+20}" y="{y+55}" font-size="20" font-weight="bold" fill="#212121">{title}</text>')
        # Bullets (Using standard dash to prevent UTF-8 encoding parser bugs)
        for idx, bullet in enumerate(bullets):
            svg_chunk.append(f'<text x="{x+20}" y="{y+85 + (idx*20)}" font-size="13" fill="#37474F">- {bullet}</text>')
        return "\n".join(svg_chunk)

    # Draw a link explanation helper
    def draw_link_box(x, y, header, lines):
        svg_chunk = []
        svg_chunk.append(f'<rect x="{x}" y="{y}" width="360" height="100" rx="8" ry="8" fill="url(#grad_link)" stroke="#FBC02D" stroke-width="1.5"/>')
        svg_chunk.append(f'<text x="{x+15}" y="{y+25}" font-size="13" font-weight="bold" fill="#F57F17">{header.upper()}</text>')
        for idx, line in enumerate(lines):
            svg_chunk.append(f'<text x="{x+15}" y="{y+48 + (idx*18)}" font-size="12.5" fill="#3E2723">{line}</text>')
        return "\n".join(svg_chunk)

    # Coordinates and content
    # NODES
    svg.append(draw_disease_node(50, 100, "1. Diabetes", "Chronic Metabolic", 
                                 ["High blood sugar damages blood vessels.", "Causes arterial stiffening and microvascular injury."], "grad_chronic"))
    
    svg.append(draw_disease_node(50, 290, "2. Congestive Heart Failure", "Chronic Cardiovascular", 
                                 ["Heart cannot pump blood efficiently.", "Causes systemic congestion and fluid retention."], "grad_chronic"))
    
    svg.append(draw_disease_node(50, 480, "3. Sepsis", "Acute Systemic Infection", 
                                 ["Severe body-wide inflammatory response.", "Leads to dramatic blood pressure collapse."], "grad_acute"))
    
    svg.append(draw_disease_node(50, 670, "4. Acute Kidney Injury", "Acute Secondary Failure", 
                                 ["Rapid loss of renal filtration capabilities.", "Spikes clinical markers (Creatinine)."], "grad_acute"))

    # CONNECTIONS & LINK EXPLANATIONS
    # Link 1: Diabetes -> CHF
    svg.append('<line x1="220" y1="230" x2="220" y2="280" stroke="#1E88E5" stroke-width="3" marker-end="url(#arrow_blue)"/>')
    svg.append(draw_link_box(450, 115, "Pathological Connection 1", 
                             ["Chronic hyperglycemia forces the heart muscle to push", "against stiffened, damaged blood vessels.", "Over time, this results in diabetic cardiomyopathy (CHF)."]))

    # Link 2: CHF -> Sepsis
    svg.append('<line x1="220" y1="420" x2="220" y2="470" stroke="#1E88E5" stroke-width="3" marker-end="url(#arrow_blue)"/>')
    svg.append(draw_link_box(450, 305, "Pathological Connection 2", 
                             ["Poor circulation causes fluid to pool in the lungs (edema)", "which acts as a breeding ground for bacterial pneumonia.", "Low physiological reserve causes it to cascade into Sepsis."]))

    # Link 3: Sepsis -> AKI
    svg.append('<line x1="220" y1="610" x2="220" y2="660" stroke="#1E88E5" stroke-width="3" marker-end="url(#arrow_blue)"/>')
    svg.append(draw_link_box(450, 495, "Pathological Connection 3", 
                             ["Sepsis triggers vasodilatory shock (severe drop in BP).", "The kidneys suffer from acute oxygen starvation (hypoperfusion).", "This leads directly to cell death (acute tubular necrosis)."]))

    # Final visual link connecting explanations together
    svg.append('<path d="M 410 165 L 410 545" fill="none" stroke="#B0BEC5" stroke-width="2" stroke-dasharray="5,5"/>')

    svg.append('</svg>')
    
    with open('../results/Disease_Cascade_Diagram.svg', 'w') as f:
        f.write("\n".join(svg))

generate_cascade_svg()
print("SUCCESS")
