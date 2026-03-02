import sys

try:
    import pynini
except ImportError:
    print("Error: 'pynini' not found. Please install it via Conda or pip (Linux/Mac only).")
    print("See README.md for installation instructions.")
    sys.exit(1)

def get_yine_g2p_rules():
    """
    Constructs and returns the Finite State Transducer (FST) for Yine 
    Grapheme-to-Phoneme (G2P) conversion.
    
    Returns:
        tuple: (main_rule_fst, cleanup_fst, awa_fst)
            - main_rule_fst: The core phonological rules.
            - cleanup_fst: Post-processing (e.g., t s -> ts).
            - awa_fst: Specific handling for 'awa' -> 'a w ɰ a'.
    """
    
    # Define character sets
    graph_vowel = pynini.union("a", "e", "i", "o", "u")
    graph_consonant = pynini.union("c", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "w", "x", "y")
    phon_vowel = pynini.union("a", "e", "i", "o", "ɨ")
    phon_consonant = pynini.union("tʃ", "h", "ŋ", "ɲ", "j", "ç", "k", "l", "m", "n", "p", "r", "d",
                                  "s", "ʃ", "t", "ts", "w", "β", "x", "c", "ɰ")
    graph_other = pynini.union("C", "G", "H", "J", "K", "L", "M", "N", "P", "R", "S", "T", "W", "X", "Y", "1",
                               "2", "3", "4", "5", "6", "7", "8", "9", "0", " ", "_")

    # Define context
    vowel = pynini.union(graph_vowel, phon_vowel)
    consonant = pynini.union(graph_consonant, phon_consonant)
    sigma_star_2 = pynini.union(vowel, consonant, graph_other).closure().optimize()

    # --- Define Individual Rules ---
    g_to_h = (pynini.cdrewrite(pynini.cross("g", "ŋ"), "", "k", sigma_star_2) @
              pynini.cdrewrite(pynini.cross("g", "ɲ"), "", "j", sigma_star_2) @
              pynini.cdrewrite(pynini.cross("g", "n"), "", "t", sigma_star_2) @
              pynini.cdrewrite(pynini.cross("g", "m"), "", "p", sigma_star_2) @
              pynini.cdrewrite(pynini.cross("g", "h"), "", "", sigma_star_2)).optimize()
              
    j_to_ç = (pynini.cdrewrite(pynini.cross("j", "ç"), "", "", sigma_star_2)).optimize()
    r_to_d = (pynini.cdrewrite(pynini.cross("r", "d"), "n", "", sigma_star_2)).optimize()
    
    csh_to_tʃ = (pynini.cdrewrite(pynini.cross("c", "tʃ"), "", "h", sigma_star_2) @
                 pynini.cdrewrite(pynini.cross("s", "ʃ"), "", "h", sigma_star_2) @
                 pynini.cdrewrite(pynini.cross("h", ""), "ʃ", "", sigma_star_2)).optimize()
                 
    u_to_ɨ = (pynini.cdrewrite(pynini.cross("u", "ɨ"), "", "", sigma_star_2)).optimize()
    w_to_β = (pynini.cdrewrite(pynini.cross("w", "β"), "", pynini.union("i", "e"), sigma_star_2)).optimize()
    x_to_c = (pynini.cdrewrite(pynini.cross("x", "c"), "", "", sigma_star_2)).optimize()
    
    y_to_ij = (pynini.cdrewrite(pynini.cross("y", "i"), consonant, consonant, sigma_star_2) @
               pynini.cdrewrite(pynini.cross("y", "j"), "", "", sigma_star_2)).optimize()
               
    space_to_us = (pynini.cdrewrite(pynini.cross(" ", "_"), "", "", sigma_star_2)).optimize()
    
    vv_to_v = (pynini.cdrewrite(pynini.cross("aa", "a"), "", "", sigma_star_2) @
              pynini.cdrewrite(pynini.cross("ee", "e"), "", "", sigma_star_2) @
              pynini.cdrewrite(pynini.cross("ii", "i"), "", "", sigma_star_2) @
              pynini.cdrewrite(pynini.cross("oo", "o"), "", "", sigma_star_2) @
              pynini.cdrewrite(pynini.cross("uu", "u"), "", "", sigma_star_2)).optimize()

    # Special Cleanup Rules
    t_s_to_ts = (pynini.cdrewrite(pynini.cross(" ", ""), "t", "ʃ", sigma_star_2) @
                 pynini.cdrewrite(pynini.cross(" ", ""), "t", "s", sigma_star_2)).optimize()
                 
    w_to_wɰ = (pynini.cdrewrite(pynini.cross("w", "ɰ"), "a", "a", sigma_star_2)).optimize()

    # --- Compose Main FST ---
    # Order matters: logic flows from left to right (or bottom to top in composition)
    phon_rules = j_to_ç @ y_to_ij @ g_to_h @ r_to_d @ csh_to_tʃ @ vv_to_v @ u_to_ɨ @ w_to_β @ x_to_c
    
    return phon_rules, t_s_to_ts, w_to_wɰ