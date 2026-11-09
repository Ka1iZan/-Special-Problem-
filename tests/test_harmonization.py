from spner.data.harmonize import project_bio_label

def test_projection_preserves_bio_prefix():
    m = {"CITY": "LOC", "PERSON": "PER", "EVENT": "IGNORE"}
    assert project_bio_label("B-CITY", m) == "B-LOC"
    assert project_bio_label("I-PERSON", m) == "I-PER"
    assert project_bio_label("B-EVENT", m) == "IGNORE"
    assert project_bio_label("O", m) == "O"
