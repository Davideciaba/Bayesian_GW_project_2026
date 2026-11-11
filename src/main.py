import argparse
import sys
import textwrap
import traceback
from pathlib import Path
from utils.py_logger import CustomLogger
from wss_point import wss_point
from bayesian_point import bayesian_point
from prior_point import plot_priors

class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter
):
    pass

def parse_args(argv=None) -> argparse.Namespace:
    """
    Parses command line arguments to configure the execution flow and I/O parameters.
    """
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=textwrap.dedent(
            """\
            LIGO-Hanford data (GW150914) analysis pipeline.
            Executes three main points of the analysis:
              1. WSS (Wide-sense stationarity) assessment and find a WSS time scale.
              2. Plots the Prior distributions for the parameters (m1, m2) and (M, q).
              3. Bayesian parameter estimation and Matched Filter SNR.
            """
        ),
    )

    parser.add_argument(
        "-out",
        "--output-dir", 
        type=Path,
        default=project_root / "Plots", 
        help="Target root directory for all outputs."
    )
    
    parser.add_argument(
        "-in",
        "--input-dir", 
        type=Path, 
        default=project_root, 
        help="Source directory containing LIGO-Hanford data (GW150914)."
    )

    parser.add_argument(
        "-wss",
        "--wss-point",
        action="store_true", 
        help="Execute first point of the analysis (WSS and time scale)."
    )

    parser.add_argument(
        "-prior",
        "--prior-distributions",
        action="store_true",
        help="Execute second point of the analysis (Plot prior distributions)."
    )

    parser.add_argument(
        "-bayes",
        "--bayesian-point",
        action="store_true",
        help="Execute third point of the analysis (Prior, Template Bank, Matched Filter SNR, and Posterior)."
    )   

    parser.add_argument(
        "-bp",
        "--bandpass",
        type=float,
        nargs=2,
        default=(20.0, 500.0),
        help="Frequency bandpass for the Bayesian analysis."
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    """
    Entry point for the LIGO-Hanford data (GW150914) pipeline.
    """
    args = parse_args(argv)

    logger = CustomLogger(name="MainLogger")
    logger.add_console_handler(level="DEBUG", use_colors=True)

    if not (args.wss_point or args.prior_distributions or args.bayesian_point):
        logger.warning("No execution flags provided. Pass -wss, -prior, and/or -bayes.")
        logger.info("Use 'python main.py -h' for usage instructions.")
        return 0
    
    active_input_dir = args.input_dir.resolve()
    active_input_dir.mkdir(parents = True, exist_ok=True)
    active_output_dir = args.output_dir.resolve()
    active_output_dir.mkdir(parents = True, exist_ok=True)
    return_code = 0
    
    try:
        logger.info(f"Global output directory mapped to: {active_output_dir}")
        logger.info(f"Global input directory mapped to: {active_input_dir}")

        if args.wss_point:
            wss_point(
                output_dir=active_output_dir,
                input_dir=active_input_dir
            )

        if args.prior_distributions:
            plot_priors(
                output_dir=active_output_dir,
                a=1.0,
                b=60.0,
                resolution=800
            )

        if args.bayesian_point:
            bayesian_point(
                output_dir=active_output_dir,
                input_dir=active_input_dir,
                bandpass = args.bandpass
            )           

    except Exception as ex:
        logger.critical(f"Unhandled main error: {str(ex)}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return_code = 1
    
    return return_code
    
if __name__ == "__main__":
    sys.exit(main())
