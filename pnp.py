# app.py
import io
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

import streamlit as st


# ------------------ Utilidades ------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\u00A0", " ")  # nbsp
    s = re.sub(r"[ \t]+", " ", s.strip())
    s_nfkd = unicodedata.normalize("NFKD", s)
    s_ascii = "".join(ch for ch in s_nfkd if not unicodedata.combining(ch))
    return s_ascii.casefold()


def pick_correct_index(options: List[str], answer_str: str) -> int:
    """
    Determina el índice correcto:
     - Si answer_str es letra A-E -> índice
     - Si es texto -> match por texto normalizado
     - Si no, -1
    """
    if not answer_str:
        return -1

    ans = answer_str.strip()
    m = re.match(r"^\s*([A-Ea-e])\s*$", ans)
    if m:
        return "ABCDE".index(m.group(1).upper())

    ans_norm = normalize_text(ans)
    for i, opt in enumerate(options):
        if normalize_text(opt) == ans_norm:
            return i

    # empieza por
    for i, opt in enumerate(options):
        if normalize_text(opt).startswith(ans_norm[:20]):
            return i

    # comparación por intersección de palabras útil si hay pequeñas diferencias
    best_i, best_score = -1, 0
    ans_words = set(ans_norm.split())
    for i, opt in enumerate(options):
        opt_words = set(normalize_text(opt).split())
        score = len(ans_words & opt_words)
        if score > best_score:
            best_score = score
            best_i = i
    if best_score > 0:
        return best_i

    return -1


# ------------------ Modelo de datos ------------------
@dataclass
class QA:
    pregunta: str
    opciones: List[str]
    correcta: int                 # índice en opciones (interno, para calificar)
    respuesta_texto: str = ""     # lo que viene despues de RESPUESTA:
    respuesta_ubicacion: str = "" # UBICACION:
    respuesta_codigo: str = ""    # CODIGO:


# ------------------ Parser PDF ------------------
def parse_pdf_to_bank(file_bytes: bytes) -> List[QA]:
    """
    Extrae preguntas desde el PDF y devuelve una lista de QA.
    Omite la primera página (carátula).
    Cada bloque se considera terminado cuando aparece una línea que inicia con 'CODIGO:'.
    """
    import pdfplumber  # requerido

    # 1) extraer todas las líneas (salteando página 0)
    full_lines: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            if i == 0:
                continue  # omitir carátula
            text = page.extract_text() or ""
            # normalizar espacios y saltos de línea
            page_lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
            for ln in page_lines:
                if ln:
                    full_lines.append(ln)

    # 2) agrupar en bloques hasta 'CODIGO:'
    blocks: List[List[str]] = []
    curr: List[str] = []
    for ln in full_lines:
        curr.append(ln)
        if re.match(r"^\s*CODIGO\s*[:\-]?", ln, flags=re.IGNORECASE):
            blocks.append(curr)
            curr = []
    # si quedó un bloque colgando, añadirlo (opcional)
    if curr:
        blocks.append(curr)

    # 3) parsear cada bloque
    qas: List[QA] = []
    for blk in blocks:
        # inicializar campos
        pregunta_lines: List[str] = []
        opciones: List[str] = []
        respuesta_texto = ""
        respuesta_ubicacion = ""
        respuesta_codigo = ""

        saw_option = False

        for ln in blk:
            # RESPUESTA:
            m_resp = re.match(r"^\s*RESPUESTA\s*[:\-]\s*(.+)$", ln, flags=re.IGNORECASE)
            if m_resp:
                respuesta_texto = m_resp.group(1).strip()
                continue

            # UBICACION:
            m_ubic = re.match(r"^\s*UBICACION\s*[:\-]\s*(.+)$", ln, flags=re.IGNORECASE)
            if m_ubic:
                respuesta_ubicacion = m_ubic.group(1).strip()
                continue

            # CODIGO:
            m_cod = re.match(r"^\s*CODIGO\s*[:\-]\s*(.+)$", ln, flags=re.IGNORECASE)
            if m_cod:
                respuesta_codigo = m_cod.group(1).strip()
                continue

            # opción que comienza con '»'
            if ln.lstrip().startswith("»"):
                saw_option = True
                opt_text = re.sub(r"^»\s*", "", ln).strip()
                opciones.append(opt_text)
                continue

            # Si todavía no vimos opciones, todo esto es parte de la pregunta
            if not saw_option:
                # si es la línea que inicia con número de pregunta, o continuación de pregunta
                if re.match(r"^\d+\.", ln) or pregunta_lines:
                    pregunta_lines.append(ln.strip())
                # si no tiene numeración pero es posible encabezado, se incluye
                else:
                    # proteger contra líneas extra (ej: titulares), pero lo agregamos por si la pregunta inicia sin numeración visible
                    pregunta_lines.append(ln.strip())
            else:
                # Si ya vimos opciones, y la línea no empieza con '»', probablemente es continuación de la última opción
                if opciones:
                    opciones[-1] = opciones[-1] + " " + ln.strip()
                else:
                    # caso raro, tratar como parte de pregunta
                    pregunta_lines.append(ln.strip())

        # construir pregunta
        pregunta = " ".join(pregunta_lines).strip()
        # quitar espacios extra y normalizar
        pregunta = re.sub(r"\s+", " ", pregunta)

        # calcular índice correcto usando respuesta_texto
        correct_idx = pick_correct_index(opciones, respuesta_texto)
        if correct_idx < 0:
            # fallback: si no encontré coincidencia, intentar buscar opción que contenga parte larga de respuesta
            ans_norm = normalize_text(respuesta_texto)
            best_i, best_score = -1, 0
            for i_opt, opt in enumerate(opciones):
                on = normalize_text(opt)
                # score: longitud de substring común
                score = len(set(ans_norm.split()) & set(on.split()))
                if score > best_score:
                    best_i, best_score = i_opt, score
            if best_i != -1 and best_score > 0:
                correct_idx = best_i
            else:
                # como último recurso, dejar 0 para no romper la app
                correct_idx = 0

        # validaciones mínimas
        if not pregunta:
            continue
        if len(opciones) < 2:
            continue

        qas.append(QA(
            pregunta=pregunta,
            opciones=opciones,
            correcta=correct_idx,
            respuesta_texto=respuesta_texto,
            respuesta_ubicacion=respuesta_ubicacion,
            respuesta_codigo=respuesta_codigo
        ))

    return qas


# ------------------ Persistencia JSON (formato solicitado) ------------------
def save_bank_to_json(bank: List[QA]) -> str:
    """
    Genera un JSON (string) con el formato exacto solicitado:
    [
      {
        "pregunta": "...",
        "opciones": [...],
        "respuesta": { "texto":"...", "ubicacion":"...", "codigo":"..." }
      },
      ...
    ]
    """
    out = []
    for q in bank:
        out.append({
            "pregunta": q.pregunta,
            "opciones": q.opciones,
            "respuesta": {
                "texto": q.respuesta_texto,
                "ubicacion": q.respuesta_ubicacion,
                "codigo": q.respuesta_codigo
            }
        })
    return json.dumps(out, ensure_ascii=False, indent=2)


def load_bank_from_json(file_bytes: bytes) -> List[QA]:
    """
    Carga el JSON en el mismo formato (pregunta/opciones/respuesta),
    y reconstruye la lista interna de QA calculando el índice correcta.
    """
    raw = json.loads(file_bytes.decode("utf-8"))
    bank: List[QA] = []
    for item in raw:
        pregunta = item.get("pregunta", "")
        opciones = item.get("opciones", []) or []
        resp = item.get("respuesta", {}) or {}
        texto = resp.get("texto", "")
        ubic = resp.get("ubicacion", "")
        cod = resp.get("codigo", "")
        correcta = pick_correct_index(opciones, texto)
        if correcta < 0:
            correcta = 0
        bank.append(QA(pregunta=pregunta, opciones=opciones, correcta=correcta,
                       respuesta_texto=texto, respuesta_ubicacion=ubic, respuesta_codigo=cod))
    return bank


# ------------------ Lógica del examen ------------------
def build_exam(bank: List[QA], n: int, seed: Optional[int] = None) -> List[QA]:
    if seed is not None:
        random.seed(seed)
    n = max(1, min(n, len(bank)))
    chosen = random.sample(bank, n)
    shuffled: List[QA] = []
    for q in chosen:
        idxs = list(range(len(q.opciones)))
        random.shuffle(idxs)
        opciones = [q.opciones[i] for i in idxs]
        correcta = idxs.index(q.correcta)
        shuffled.append(QA(
            pregunta=q.pregunta,
            opciones=opciones,
            correcta=correcta,
            respuesta_texto=q.respuesta_texto,
            respuesta_ubicacion=q.respuesta_ubicacion,
            respuesta_codigo=q.respuesta_codigo
        ))
    return shuffled


# ------------------ Interfaz Streamlit ------------------
st.set_page_config(page_title="Simulador de Examen", page_icon="📝", layout="wide")
st.title("📝 Simulador de Examen desde PDF/JSON")
st.caption("Carga tu PDF (con 'RESPUESTA:', 'UBICACION:' y 'CODIGO:') o un JSON ya procesado. Omite carátula y descarga banco en el formato solicitado.")

with st.sidebar:
    st.header("📥 Fuente de datos")
    file_pdf = st.file_uploader("Sube PDF (recomendado)", type=["pdf"])
    file_json = st.file_uploader("…o JSON (banco ya procesado)", type=["json"])
    n_preg = st.number_input("Número de preguntas por examen", 10, 200, 100, 10)
    seed = st.number_input("Semilla aleatoria (opcional)", min_value=0, max_value=999999, value=0, step=1)
    seed_val = int(seed) if seed != 0 else None
    parse_btn = st.button("Procesar / Cargar banco")

# Estado
if "bank" not in st.session_state:
    st.session_state.bank: List[QA] = []
if "exam" not in st.session_state:
    st.session_state.exam: List[QA] = []
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# Procesar / Cargar
if parse_btn:
    try:
        if file_json is not None:
            st.session_state.bank = load_bank_from_json(file_json.read())
            st.success(f"✅ Banco cargado desde JSON: {len(st.session_state.bank)} preguntas.")
        elif file_pdf is not None:
            st.info("⏳ Extrayendo preguntas del PDF… esto puede tardar según el tamaño.")
            bank = parse_pdf_to_bank(file_pdf.read())
            st.session_state.bank = bank
            st.success(f"✅ Banco extraído del PDF: {len(bank)} preguntas.")
            # Ofrecer descarga del JSON resultante (en el formato solicitado)
            json_str = save_bank_to_json(st.session_state.bank)
            st.download_button("💾 Descargar banco como JSON", data=json_str.encode("utf-8"),
                               file_name="banco_preguntas.json", mime="application/json")
        else:
            st.warning("Sube un PDF o un JSON para continuar.")
    except Exception as e:
        st.error(f"Error al procesar: {e}")

# Construcción del examen
colL, colR = st.columns([2, 1])
with colL:
    st.header("🧪 Examen")
    if st.session_state.bank:
        if st.button("🎲 Iniciar nuevo examen"):
            st.session_state.exam = build_exam(st.session_state.bank, n_preg, seed_val)
            st.session_state.answers = {}
            st.session_state.submitted = False

        if st.session_state.exam:
            for i, q in enumerate(st.session_state.exam, start=1):
                st.markdown(f"**{i}. {q.pregunta}**")
                st.session_state.answers[i] = st.radio(
                    "Elige una opción",
                    q.opciones,
                    index=None,
                    key=f"q_{i}",
                    label_visibility="collapsed"
                )
                st.divider()

            if not st.session_state.submitted:
                if st.button("✅ Enviar examen"):
                    st.session_state.submitted = True
        else:
            st.info("Presiona **Iniciar nuevo examen** para comenzar.")
    else:
        st.info("Primero carga y procesa tu PDF o JSON en la barra lateral.")

with colR:
    st.header("📊 Resultados")
    if st.session_state.exam and st.session_state.submitted:
        total = len(st.session_state.exam)
        correct = 0
        review_rows = []
        for i, q in enumerate(st.session_state.exam, start=1):
            sel = st.session_state.answers.get(i)
            correct_text = q.opciones[q.correcta]
            is_ok = (sel == correct_text)
            correct += int(is_ok)
            review_rows.append((i, "✅" if is_ok else "❌", sel or "—", correct_text))

        st.metric("Puntaje", f"{correct}/{total}")
        st.progress(correct / total if total else 0.0)

        st.subheader("🔍 Revisión rápida")
        for (i, mark, sel, corr) in review_rows:
            st.write(f"**{i}.** {mark} • Tu respuesta: _{sel}_ • Correcta: **{corr}**")

        # Botones para repetir o exportar
        if st.button("🔁 Reintentar (nuevo examen)"):
            st.session_state.exam = build_exam(st.session_state.bank, n_preg, seed_val)
            st.session_state.answers = {}
            st.session_state.submitted = False

        # Descargar banco (formato solicitado)
        json_str = save_bank_to_json(st.session_state.bank)
        st.download_button("💾 Descargar banco (JSON)", data=json_str.encode("utf-8"),
                           file_name="banco_preguntas.json", mime="application/json")
    else:
        st.caption("Aquí verás tu puntaje y revisión cuando envíes el examen.")
