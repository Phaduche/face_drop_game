import cv2

from face_game import CANVAS_SIZE, make_demo_portrait, prepare_face_game


def test_demo_portrait_and_pieces_are_valid():
    portrait = make_demo_portrait()
    assert portrait.shape == (900, 900, 3)
    assert portrait.dtype.name == "uint8"

    board_uri, pieces, metadata = prepare_face_game(portrait)

    assert board_uri.startswith("data:image/png;base64,")
    assert len(pieces) == 4
    assert {piece["key"] for piece in pieces} == {
        "left_eye",
        "right_eye",
        "nose",
        "mouth",
    }
    assert isinstance(metadata["used_fallback"], bool)

    for piece in pieces:
        assert piece["src"].startswith("data:image/png;base64,")
        assert 0 <= piece["targetX"] < CANVAS_SIZE
        assert 0 <= piece["targetY"] < CANVAS_SIZE
        assert piece["w"] > 0
        assert piece["h"] > 0


def test_center_crop_fallback_handles_non_face_image():
    blank = cv2.cvtColor(
        make_demo_portrait(300) * 0,
        cv2.COLOR_BGR2RGB,
    )
    board_uri, pieces, metadata = prepare_face_game(blank)

    assert board_uri.startswith("data:image/png;base64,")
    assert len(pieces) == 4
    assert metadata["used_fallback"] is True
