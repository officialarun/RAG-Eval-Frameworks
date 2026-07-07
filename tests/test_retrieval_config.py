from raglens.config import DEFAULT_CONFIG, RetrievalConfig


def test_default_config_flags_known_bad_sections():
    assert DEFAULT_CONFIG.is_bad_section("References")
    assert DEFAULT_CONFIG.is_bad_section("Bibliography")
    assert DEFAULT_CONFIG.is_bad_section("List of Tables")


def test_default_config_allows_normal_sections():
    assert not DEFAULT_CONFIG.is_bad_section("Introduction")
    assert not DEFAULT_CONFIG.is_bad_section("Stochastic Gradient Descent")


def test_is_bad_section_normalizes_punctuation_and_case():
    assert DEFAULT_CONFIG.is_bad_section("REFERENCES:")
    assert DEFAULT_CONFIG.is_bad_section("  references  ")


def test_custom_config_overrides_bad_sections():
    custom = RetrievalConfig(bad_sections=frozenset({"appendix"}))
    assert custom.is_bad_section("Appendix A")
    assert not custom.is_bad_section("References")


def test_default_config_is_a_shared_singleton_instance():
    assert DEFAULT_CONFIG.default_top_k == 5
    assert DEFAULT_CONFIG.exclude_reference_sections is True
