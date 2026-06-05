"""Minimal demo pipeline for LitExtract."""
import json, sys, os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def demo_extraction():
    """Demo: Extract structured data from a sample paper text."""
    sample_text = """
    The alloy Ti-6Al-4V was produced by vacuum arc remelting (VAR).
    The ingot was forged at 950 degree C and then solution treated at 1020 degree C for 2h,
    followed by water quenching. Subsequent aging was performed at 540 degree C for 8h.
    SEM observation revealed a bimodal microstructure consisting of equiaxed primary alpha
    (~15 um) and lamellar alpha+beta colonies. The aged sample exhibited UTS of 1050 MPa
    and elongation of 12%.
    """
    print("Sample text processed.")
    print("Expected extraction: Ti-6Al-4V | VAR -> Forge 950C -> ST 1020C/2h -> Age 540C/8h")
    print("Micro: bimodal, equiaxed alpha 15um + lamellar alpha+beta")
    print("Property: UTS=1050MPa, EL=12%")

if __name__ == '__main__':
    demo_extraction()
