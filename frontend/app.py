from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd
import streamlit as st

from utils.checkpoint import load_bundle
from utils.data_io import append_activity_row, load_activity_log
from utils.insights import describe_insights
from utils.next_category import next_categories
from utils.recommendation import suggest_tasks_for_category
from utils.recommendation_log import RecommendationLogRow, append_recommendation_log


def _safe_path(text: str) -> Path:
    text = (text or "").strip()
    if not text:
        return Path()
    return Path(text)


def _copy_example_log(dest: Path) -> None:
    src = Path("data/activity_log_example.csv")
    if not src.exists():
        raise FileNotFoundError(f"Example log not found: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)


def _data_maturity_label(n_rows: int) -> tuple[str, float]:
    if n_rows < 10:
        return "faible", 0.25
    if n_rows < 30:
        return "moyenne", 0.5
    if n_rows < 100:
        return "bonne", 0.75
    return "elevee", 0.9


def _pick_best_hour(hourly):
    if not hourly:
        return None
    return max(hourly, key=lambda h: (h.completion_rate, h.n))


def _pick_best_category(categories, min_samples: int = 2):
    eligible = [c for c in categories if c.n >= min_samples]
    if not eligible:
        return None
    return max(eligible, key=lambda c: (c.completion_rate, c.n))


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.0f}%"


def _format_minutes(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.0f} min"


def _format_age(last_seen: datetime | None, max_ts: datetime | None) -> str:
    if last_seen is None or max_ts is None:
        return "n/a"
    minutes_ago = max(0, int((max_ts - last_seen).total_seconds() / 60.0))
    if minutes_ago < 60:
        return f"{minutes_ago} min"
    if minutes_ago < 24 * 60:
        return f"{minutes_ago // 60} h"
    return f"{minutes_ago // (24 * 60)} j"


def _pick_best_task(suggestions: list[dict[str, object]], available_minutes: float | None):
    if not suggestions:
        return None
    if available_minutes is None or available_minutes <= 0:
        return suggestions[0]
    limit = float(available_minutes) * 1.2
    for s in suggestions:
        if float(s.get("avg_duration_min", 0.0)) <= limit:
            return s
    return suggestions[0]


@st.cache_data(show_spinner=False)
def _load_activity_df(path: str, mtime: float) -> pd.DataFrame:
    return load_activity_log(path).df


@st.cache_data(show_spinner=False)
def _describe_insights_cached(path: str, mtime: float) -> dict[str, object]:
    df = _load_activity_df(path, mtime)
    return describe_insights(df)


@st.cache_resource(show_spinner=False)
def _load_model_bundle(path: str, mtime: float):
    return load_bundle(path, map_location="cpu")


def _render_insights(df: pd.DataFrame, *, insights: dict[str, object] | None = None) -> None:
    insights = insights or describe_insights(df)

    completion = insights.get("overall_completion_rate")
    avg_energy = insights.get("overall_avg_energy")
    categories = insights.get("category", [])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Taux de reussite", _format_pct(completion))
    with col2:
        st.metric("Energie moyenne", "n/a" if avg_energy is None or pd.isna(avg_energy) else f"{float(avg_energy):.2f}")
    with col3:
        st.metric("Nb categories", f"{len(categories)}")

    hourly = insights.get("hourly", [])
    if hourly:
        hdf = pd.DataFrame([asdict(h) for h in hourly]).sort_values("hour")
        hdf["completion_rate_pct"] = hdf["completion_rate"] * 100.0
        st.subheader("Reussite par heure")
        st.bar_chart(hdf.set_index("hour")[["completion_rate_pct"]])
        if "avg_energy" in hdf.columns and hdf["avg_energy"].notna().any():
            st.subheader("Energie moyenne par heure")
            st.line_chart(hdf.set_index("hour")[["avg_energy"]])

    low_hours = insights.get("low_productivity_hours", [])
    if low_hours:
        st.subheader("Heures a risque (interruptions)")
        st.table(pd.DataFrame([asdict(h) for h in low_hours]))

    if categories:
        st.subheader("Resume par categorie")
        cdf = pd.DataFrame([asdict(c) for c in categories])
        cdf["completion_rate_pct"] = cdf["completion_rate"] * 100.0
        st.dataframe(
            cdf[["category", "n", "completion_rate_pct", "avg_duration_min", "avg_energy"]].sort_values(
                ["n", "completion_rate_pct"], ascending=[False, False]
            ),
            width="stretch",
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(page_title="LocalBrain", layout="wide")
    st.title("LocalBrain - Assistant de focus")
    st.caption("Votre journal devient une prochaine action claire. 100% local, rien ne sort de votre machine.")
    st.markdown("**Ce que vous obtenez**")
    st.markdown(
        "- Une recommandation simple pour savoir quoi faire maintenant.\n"
        "- Des raisons claires basees sur votre historique.\n"
        "- Des insights pour mieux planifier vos moments forts."
    )

    st.session_state.setdefault("last_recommendation", None)

    data_path_default = "data/activity_log.csv"
    model_path_default = "artifacts/productivity_model.pt"
    data_path = data_path_default
    model_path = model_path_default
    seq_len_override = 0
    log_recommendation = False
    model_status_slot = None

    with st.sidebar:
        st.header("Votre contexte")
        energy = st.slider("Energie actuelle (0 = inconnue)", min_value=0, max_value=5, value=0, step=1)
        available_minutes = st.number_input(
            "Temps dispo (minutes)", min_value=0, max_value=600, value=0, step=5
        )
        top_k = st.slider("Nombre de suggestions", min_value=1, max_value=10, value=5)

        advanced = st.expander("Options avancees")
        with advanced:
            data_path = st.text_input("Chemin du journal", value=data_path_default)
            model_path = st.text_input("Chemin du modele (optionnel)", value=model_path_default)
            seq_len_override = st.number_input(
                "Seq len override (0=auto)", min_value=0, max_value=128, value=0, step=1
            )
            log_recommendation = st.checkbox("Enregistrer la recommandation", value=False)
            st.divider()
            model_status_slot = st.container()

    data_p = _safe_path(data_path)
    model_p = _safe_path(model_path)
    current_energy = int(energy) if int(energy) in {1, 2, 3, 4, 5} else None
    current_minutes = float(available_minutes) if float(available_minutes) > 0 else None

    if model_status_slot is not None:
        with model_status_slot:
            st.subheader("Etat du modele")
            if model_p and model_p.is_file():
                model_mtime = model_p.stat().st_mtime
                st.success("Modele trouve")
                if st.checkbox("Afficher les details du modele", value=False):
                    try:
                        bundle = _load_model_bundle(str(model_p), model_mtime)
                        st.write(f"Cree: `{bundle.extra.get('created_at_utc', 'unknown')}`")
                        st.write(f"Seq len: `{bundle.extra.get('seq_len', 'unknown')}`")
                        st.write(f"Categories: `{len(bundle.vocab.category.id_to_token)}`")
                    except Exception as e:
                        st.warning("Echec du chargement du modele.")
                        st.code(str(e))
            else:
                st.info("Mode heuristique (pas de modele trouve)")

    if (not data_p.exists()) or data_p.is_dir():
        if data_p.is_dir():
            data_p = Path(data_path_default)
        st.subheader("Demarrage rapide")
        st.warning("Aucun journal trouve. Creez un exemple ou importez un CSV.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Option 1: utiliser un exemple**")
            st.caption("Cree un petit journal pour tester en 30 secondes.")
            if st.button("Creer un journal exemple", type="primary"):
                try:
                    _copy_example_log(data_p)
                except Exception as e:
                    st.error("Impossible de creer le journal exemple.")
                    st.code(str(e))
                else:
                    st.success("Journal exemple cree. Rechargement...")
                    st.rerun()
        with col2:
            st.markdown("**Option 2: importer un CSV**")
            uploaded = st.file_uploader("Importer un CSV", type=["csv"])
            if uploaded is not None:
                data_p.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = data_p.parent / "_uploaded_activity_log.csv"
                tmp_path.write_bytes(uploaded.getvalue())
                try:
                    tmp_log = load_activity_log(tmp_path)
                except Exception as e:
                    st.error("CSV invalide.")
                    st.code(str(e))
                else:
                    st.success("CSV valide. Apercu ci-dessous.")
                    st.dataframe(tmp_log.df.head(20), width="stretch", hide_index=True)
                    if st.button("Enregistrer ce journal"):
                        tmp_path.replace(data_p)
                        st.success(f"Journal enregistre dans {data_p}")
                        st.rerun()

        st.stop()

    try:
        data_mtime = data_p.stat().st_mtime
        df = _load_activity_df(str(data_p), data_mtime)
    except Exception as e:
        st.error("Impossible de charger le journal.")
        st.code(str(e))
        return

    max_ts = None if len(df) == 0 else df["timestamp_dt"].max()
    max_ts_str = "n/a" if max_ts is None or pd.isna(max_ts) else max_ts.isoformat(sep=" ")

    insights = _describe_insights_cached(str(data_p), data_mtime)
    completion = insights.get("overall_completion_rate")
    avg_energy = insights.get("overall_avg_energy")
    hourly = insights.get("hourly", [])
    cstats = insights.get("category", [])

    best_hour = _pick_best_hour(hourly)
    best_hour_label = "n/a" if best_hour is None else f"{int(best_hour.hour):02d}h"
    best_cat = _pick_best_category(cstats)
    best_cat_label = "n/a" if best_cat is None else str(best_cat.category)
    maturity_label, maturity_score = _data_maturity_label(len(df))

    st.subheader("Valeur en un coup d'oeil")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Taux de reussite", _format_pct(completion))
    with col2:
        st.metric("Energie moyenne", "n/a" if avg_energy is None or pd.isna(avg_energy) else f"{float(avg_energy):.2f}")
    with col3:
        st.metric("Meilleure heure", best_hour_label)
    with col4:
        st.metric("Categorie forte", best_cat_label)

    st.caption(f"Maturite du journal: {maturity_label} (plus de donnees = meilleures recommandations)")
    st.progress(maturity_score)

    if len(df) < 10:
        st.info("Ajoutez 10 a 20 lignes pour des recommandations plus fiables.")

    tabs = st.tabs(["Mon prochain focus", "Journal", "Analyses"])

    with tabs[0]:
        st.subheader("Votre prochaine action")
        st.caption("Cliquez pour obtenir une recommandation claire, sans jargon.")

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            recommend_clicked = st.button("Generer ma prochaine action", type="primary")
        with btn_col2:
            if st.button("Effacer", type="secondary"):
                st.session_state["last_recommendation"] = None

        if recommend_clicked:
            bundle = None
            if model_p and model_p.is_file():
                try:
                    bundle = _load_model_bundle(str(model_p), model_p.stat().st_mtime)
                except Exception as e:
                    st.warning("Modele indisponible. Passage en mode heuristique.")
                    st.code(str(e))
            topk, source = next_categories(
                df,
                model_path=(str(model_p) if model_p and model_p.is_file() else None),
                bundle=bundle,
                k=int(top_k),
                seq_len_override=int(seq_len_override),
            )

            if not topk:
                st.warning("Pas assez de donnees pour recommander. Ajoutez quelques lignes.")
            else:
                best_category = str(topk[0][0])
                suggestions = suggest_tasks_for_category(
                    df=df,
                    category=best_category,
                    current_energy=current_energy,
                    available_minutes=current_minutes,
                    top_n=5,
                )

                st.session_state["last_recommendation"] = {
                    "generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
                    "source": source,
                    "topk": topk,
                    "best_category": best_category,
                    "suggestions": [
                        {
                            "task_name": s.task_name,
                            "score": s.score,
                            "completion_rate": s.completion_rate,
                            "avg_duration_min": s.avg_duration_min,
                            "last_seen": "" if s.last_seen is None else s.last_seen.isoformat(sep=" "),
                        }
                        for s in suggestions
                    ],
                    "n_rows": int(len(df)),
                    "last_ts": max_ts_str,
                }

                if log_recommendation:
                    rec_path = data_p.parent / "recommendation_log.csv"
                    append_recommendation_log(
                        rec_path,
                        RecommendationLogRow(
                            timestamp=datetime.now(),
                            energy_level=current_energy,
                            available_minutes=current_minutes,
                            topk_categories=topk,
                            model_path=(str(model_p) if model_p and model_p.is_file() else None),
                        ),
                    )
                    st.success(f"Recommandation enregistree dans {rec_path}")

        rec = st.session_state.get("last_recommendation")
        if rec:
            if rec.get("n_rows") != int(len(df)) or rec.get("last_ts") != max_ts_str:
                st.warning("Le journal a change depuis la recommandation. Regenerer pour etre a jour.")

            source_label = "modele" if rec.get("source") == "model" else "heuristique"
            st.caption(f"Genere: {rec.get('generated_at')} | Source: {source_label}")

            best_category = rec.get("best_category")
            if best_category:
                st.success(f"Categorie conseillee: {best_category}")

            topk = rec.get("topk") or []
            if topk:
                rec_df = pd.DataFrame(topk, columns=["categorie", "score"])
                rec_df["probabilite"] = (rec_df["score"].astype(float) * 100.0).round(1).astype(str) + "%"
                st.subheader("Categories recommandees")
                st.dataframe(rec_df[["categorie", "probabilite"]], width="stretch", hide_index=True)

            suggestions = rec.get("suggestions") or []
            best_task = _pick_best_task(suggestions, current_minutes)
            if best_task:
                last_seen_raw = str(best_task.get("last_seen") or "")
                last_seen_dt = datetime.fromisoformat(last_seen_raw) if last_seen_raw else None
                age_label = _format_age(last_seen_dt, max_ts)

                st.subheader("Action conseillee")
                st.write(f"**{best_task.get('task_name', '')}**")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Taux de reussite", _format_pct(best_task.get("completion_rate")))
                with c2:
                    st.metric("Duree moyenne", _format_minutes(best_task.get("avg_duration_min")))
                with c3:
                    st.metric("Dernier passage", age_label)

                reasons: list[str] = []
                completion_rate = float(best_task.get("completion_rate") or 0.0)
                avg_dur = float(best_task.get("avg_duration_min") or 0.0)
                if completion_rate >= 0.6:
                    reasons.append("Bon taux de reussite dans votre historique.")
                if current_minutes is not None and current_minutes > 0:
                    if abs(avg_dur - current_minutes) <= max(10.0, current_minutes * 0.3):
                        reasons.append("Duree proche de votre temps dispo.")
                if last_seen_dt is not None and max_ts is not None:
                    minutes_ago = (max_ts - last_seen_dt).total_seconds() / 60.0
                    if minutes_ago >= 120:
                        reasons.append("Pas faite recemment.")

                if reasons:
                    st.markdown("**Pourquoi cette action ?**\n" + "\n".join(f"- {r}" for r in reasons))
                else:
                    st.caption("Pourquoi: choix base sur votre historique recent.")

            if suggestions:
                rows = []
                for s in suggestions:
                    last_seen_raw = str(s.get("last_seen") or "")
                    last_seen_dt = datetime.fromisoformat(last_seen_raw) if last_seen_raw else None
                    rows.append(
                        {
                            "tache": s.get("task_name"),
                            "taux_reussite": _format_pct(s.get("completion_rate")),
                            "duree_moy": _format_minutes(s.get("avg_duration_min")),
                            "dernier": _format_age(last_seen_dt, max_ts),
                        }
                    )
                st.subheader("Autres taches possibles")
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.info("Ajoutez des taches dans ce type pour des suggestions plus precises.")
        else:
            st.info("Cliquez sur 'Generer ma prochaine action' pour obtenir une recommandation.")

    with tabs[1]:
        st.subheader("Journal")
        st.caption(f"{len(df)} lignes | derniere activite: {max_ts_str}")

        with st.form("add_activity"):
            ts = st.date_input("Date", value=datetime.now().date())
            tm = st.time_input("Heure", value=datetime.now().time().replace(microsecond=0))
            task_name = st.text_input("Nom de la tache")
            task_category = st.text_input("Categorie")
            duration_min = st.number_input(
                "Duree (minutes)", min_value=1.0, max_value=1440.0, value=25.0, step=5.0
            )
            energy_level = st.selectbox("Niveau d'energie (optionnel)", options=[""] + [1, 2, 3, 4, 5], index=0)
            outcome_label = st.selectbox("Resultat", options=["termine", "interrompu"], index=0)
            submitted = st.form_submit_button("Ajouter au journal")

        if submitted:
            if not task_name.strip() or not task_category.strip():
                st.warning("Merci de renseigner la tache et la categorie.")
            else:
                dt = datetime.combine(ts, tm)
                outcome = "completed" if outcome_label == "termine" else "interrupted"
                append_activity_row(
                    data_p,
                    timestamp=dt,
                    task_name=task_name.strip(),
                    task_category=task_category.strip(),
                    duration_min=float(duration_min),
                    energy_level=(int(energy_level) if isinstance(energy_level, int) else None),
                    outcome=outcome,
                )
                st.session_state["last_recommendation"] = None
                st.success("Activite ajoutee. Rechargement...")
                st.rerun()

        st.subheader("Dernieres activites")
        categories = sorted(df["task_category"].astype(str).unique().tolist())
        sel_categories = st.multiselect("Filtrer par categorie", options=categories, default=categories)
        outcome_map = {"termine": "completed", "interrompu": "interrupted"}
        sel_outcomes_labels = st.multiselect(
            "Filtrer par resultat", options=list(outcome_map.keys()), default=list(outcome_map.keys())
        )
        sel_outcomes = [outcome_map[o] for o in sel_outcomes_labels]

        filtered = df.copy()
        if sel_categories:
            filtered = filtered[filtered["task_category"].astype(str).isin(sel_categories)]
        if sel_outcomes:
            filtered = filtered[filtered["outcome"].astype(str).isin(sel_outcomes)]

        st.dataframe(
            filtered[["timestamp", "task_name", "task_category", "duration_min", "energy_level", "outcome"]]
            .tail(200),
            width="stretch",
            hide_index=True,
        )

        csv_bytes = filtered[["timestamp", "task_name", "task_category", "duration_min", "energy_level", "outcome"]]
        csv_bytes = csv_bytes.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Telecharger le CSV filtre", data=csv_bytes, file_name="activity_log_filtered.csv", mime="text/csv"
        )

    with tabs[2]:
        st.subheader("Analyses")
        _render_insights(df, insights=insights)

    with st.expander("Mode avance (optionnel)"):
        st.caption("Pour entrainer le modele local (optionnel):")
        st.code("python -m training.train --data data/activity_log.csv --out artifacts/productivity_model.pt")


if __name__ == "__main__":
    main()
