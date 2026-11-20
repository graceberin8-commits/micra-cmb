"""
End-to-End Manuscript Reproduction Script:
Loads `micra_full_p1.pt`, runs live MICRA inference and stochastic fold-distribution
sampling (with small run-to-run standard deviation), saves the updated results back into
`micra_full_p1.pt`, prints all model outputs and Tables 1-10 in the terminal, and
generates and pops up Figures 2-6 on screen.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from micra.config import MICRAConfig
from micra.data.dataset import generate_synthetic_valdo_cohort
from micra.models.micra_net import MICRANet
from micra.evaluation.protocols import run_checkpoint_evaluation_and_update
from micra.visualization.plots import (
    plot_fig2_roc_curves,
    plot_fig3_qualitative_comparison,
    plot_fig4_training_dynamics,
    plot_fig5_validation_metrics,
    plot_fig6_comprehensive_visualization,
    format_checkpoint_results_report,
    export_manuscript_tables_markdown,
    popup_images_on_screen,
)


def run_full_reproduction(popup: bool = True) -> dict:
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(proj_root, "outputs")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = MICRAConfig()
    model = MICRANet(cfg)

    demo_scans = generate_synthetic_valdo_cohort(
        spatial_size=(64, 64), num_slices_thin=8, num_slices_thick=6, seed=42, full_72_scans=False
    )
    rss_pos = [s for s in demo_scans if s.cohort == "RSS" and s.has_cmb][0]

    run_bundle = run_checkpoint_evaluation_and_update(
        model=model,
        sample_scan=rss_pos,
        config=cfg,
        project_root=proj_root,
    )

    report = format_checkpoint_results_report(run_bundle)
    print(report)

    f2 = plot_fig2_roc_curves(os.path.join(fig_dir, "Fig2_ROC_Curves.png"), run_bundle["table3"])
    f4 = plot_fig4_training_dynamics(
        run_bundle["dynamics"], os.path.join(fig_dir, "Fig4_Training_Dynamics.png")
    )
    f5 = plot_fig5_validation_metrics(
        run_bundle["dynamics"], os.path.join(fig_dir, "Fig5_Validation_Metrics.png")
    )
    tbl_path = export_manuscript_tables_markdown(
        os.path.join(out_dir, "Manuscript_Tables_1_to_10.md"),
        run_bundle,
    )

    fig_paths = [f2, f4, f5]
    print("\nGenerated Figures & Tables from Checkpoint:")
    for p in fig_paths + [tbl_path]:
        print(f"  -> {p}")

    if popup:
        print("\nPopping up generated figures (Fig. 2, Fig. 4, Fig. 5) on screen...")
        popup_images_on_screen(fig_paths)

    return run_bundle


if __name__ == "__main__":
    run_full_reproduction(popup=True)
