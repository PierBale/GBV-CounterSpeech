"""PDF conversion using the hosted LlamaParse API (LlamaCloud SDK)."""

import hashlib
import json
import os

from .config import PROJECT_ROOT, CardCreationConfig
from .progress import report


def parse_pdfs(config: CardCreationConfig, overwrite: bool = False) -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    report("parse", f"Ricerca PDF in {config.pdf_dir}")
    if not config.pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {config.pdf_dir}")
    paths = sorted(
        path for path in config.pdf_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not paths:
        raise ValueError(f"No PDFs found in {config.pdf_dir}")
    report("parse", f"Trovati {len(paths)} PDF; destinazione Markdown: {config.markdown_dir}")
    if overwrite:
        report("parse", "Overwrite attivo: i Markdown esistenti verranno rigenerati")

    client = None
    failures = []
    parsed = 0
    reused = 0
    try:
        for position, pdf in enumerate(paths, start=1):
            relative = pdf.relative_to(config.pdf_dir)
            report("parse", f"PDF {position}/{len(paths)}: {relative}; verifica cache")
            output = config.markdown_dir / relative.with_suffix(".md")
            sidecar = output.with_suffix(".parse.json")
            signature = {
                "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                "tier": config.llama_tier,
                "version": config.llama_version,
                "base_url": config.llama_base_url,
            }
            if output.exists() and output.stat().st_size and not overwrite:
                if sidecar.exists():
                    cached = json.loads(sidecar.read_text(encoding="utf-8"))
                    if any(cached.get(key) != value for key, value in signature.items()):
                        raise ValueError(
                            f"Stale Markdown cache for {pdf}; use parse --overwrite."
                        )
                else:
                    report("parse", "Manifest assente: provenienza del Markdown non verificabile")
                reused += 1
                report("parse", f"Markdown riutilizzato, nessuna chiamata LlamaParse: {output}")
                continue

            if client is None:
                from llama_cloud import LlamaCloud

                api_key = os.environ.get(config.llama_api_key_env)
                if not api_key:
                    raise RuntimeError(
                        f"Set {config.llama_api_key_env} before calling LlamaParse."
                    )
                report(
                    "parse", f"Connessione LlamaCloud: {config.llama_base_url}; "
                    f"tier={config.llama_tier}, versione={config.llama_version}",
                )
                client = LlamaCloud(api_key=api_key, base_url=config.llama_base_url)

            try:
                report("parse", f"Upload PDF {position}/{len(paths)}: {relative}")
                uploaded = client.files.create(file=str(pdf), purpose="parse")
                report("parse", f"Parsing remoto in corso: {relative}")
                result = client.parsing.parse(
                    file_id=uploaded.id,
                    tier=config.llama_tier,
                    version=config.llama_version,
                    expand=["markdown_full"],
                )
                markdown = result.markdown_full or ""
                if not markdown.strip():
                    raise ValueError("LlamaParse returned empty Markdown.")
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(".md.tmp")
                temporary.write_text(markdown, encoding="utf-8")
                temporary.replace(output)
                sidecar.write_text(
                    json.dumps(
                        {**signature, "pdf_file": relative.as_posix(),
                         "parse_job_id": result.job.id},
                        ensure_ascii=False, indent=2,
                    ) + "\n",
                    encoding="utf-8",
                )
                parsed += 1
                report("parse", f"PDF {position}/{len(paths)} completato; Markdown salvato: {output}")
            except Exception as error:
                failures.append(f"{relative}: {error}")
                report("parse", f"ERRORE su {relative}: {error}")
    finally:
        if client is not None:
            client.close()

    report("parse", f"Riepilogo: {parsed} convertiti, {reused} riutilizzati, {len(failures)} falliti")
    if failures:
        raise RuntimeError("LlamaParse failures:\n" + "\n".join(failures))
