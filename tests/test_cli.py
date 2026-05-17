from pdf_md.cli import build_parser, main


def test_parser_defaults():
    args = build_parser().parse_args(["some.pdf"])
    assert args.source == "some.pdf"
    assert args.format == "markdown"
    assert args.use_ocr is True


def test_parser_no_ocr_and_format():
    args = build_parser().parse_args(["x.pdf", "--no-ocr", "--format", "text"])
    assert args.use_ocr is False
    assert args.format == "text"


def test_main_runs_single_pdf(sample_pdf, capsys):
    rc = main([str(sample_pdf), "--no-ocr", "--workers", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert len(out.strip()) > 0   # markdown printed to stdout
