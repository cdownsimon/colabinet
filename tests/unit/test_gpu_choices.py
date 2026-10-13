"""Which GPU classes the operator may create a session on.

`gpu_choices` is the single source of truth behind both the /new form and the command
palette, so a disabled class cannot appear in one and not the other. A class is matched on
its family ("A100" also enables "A100-80"), and CPU is always kept as the fallback.
"""

from colabinet.web.views import gpu_choices


def test_default_setting_enables_every_configured_family():
    # The shipped default is "G4,A100,L4,T4", which covers all four GPU families.
    assert gpu_choices("G4,A100,L4,T4") == ["CPU", "T4", "T4-HIGHMEM", "L4", "A100", "A100-80", "G4"]


def test_family_variants_follow_their_base_class():
    assert gpu_choices("A100") == ["CPU", "A100", "A100-80"]
    assert gpu_choices("T4") == ["CPU", "T4", "T4-HIGHMEM"]


def test_disabling_a_family_removes_its_variants_too():
    assert gpu_choices("L4,G4") == ["CPU", "L4", "G4"]
    assert "A100-80" not in gpu_choices("L4,G4")
    assert "T4-HIGHMEM" not in gpu_choices("L4,G4")


def test_cpu_is_available_even_when_no_gpu_is_eligible():
    # Unchecking every box in Settings is a reachable state, not a malformed one.
    assert gpu_choices("") == ["CPU"]
    assert gpu_choices("CPU") == ["CPU"]


def test_entries_are_normalised_and_unknown_classes_ignored():
    assert gpu_choices(" a100 , l4 ") == ["CPU", "L4", "A100", "A100-80"]
    assert gpu_choices("H100,B200") == ["CPU"]


def test_order_is_stable_so_the_form_and_menu_render_alike():
    assert gpu_choices("G4,A100") == gpu_choices("A100,G4") == ["CPU", "A100", "A100-80", "G4"]
