import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "generation",
            "gold",
            "case_canonicalization",
            "coverage",
            "global_canonicalization",
            "profiles",
            "entropy",
            "distribution",
            "all",
        ],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of examples to process per dataset adapter"
    )

    parser.add_argument(
        "--suffix",
        default="",
        help="Label appended to RESULT tables (coverage/entropy/profiles/"
             "matrix/js), e.g. '_500'. Intermediates are never suffixed."
    )

    args = parser.parse_args()


    if args.stage in [
        "generation",
        "all",
    ]:
        from generation import generate_all

        print(
            "\n=== Generating model answers ==="
        )

        generate_all(limit=args.limit)



    if args.stage in [
        "gold",
        "all",
    ]:
        from extract_gold_features import (
            extract_gold_features
        )

        print(
            "\n=== Extracting gold features ==="
        )

        extract_gold_features(limit=args.limit)



    if args.stage in [
        "case_canonicalization",
        "all",
    ]:

        from canonicalize_case import (
            canonicalize_cases
        )

        print(
            "\n=== Canonicalizing cases ==="
        )

        canonicalize_cases()



    if args.stage in [
        "coverage",
        "all",
    ]:

        from coverage import (
            compute_coverage
        )

        print(
            "\n=== Computing coverage ==="
        )

        compute_coverage(suffix=args.suffix)



    if args.stage in [
        "global_canonicalization",
        "all",
    ]:

        from canonicalize_global import (
            canonicalize_global
        )

        print(
            "\n=== Building global feature space ==="
        )

        canonicalize_global()



    if args.stage in [
        "profiles",
        "all",
    ]:

        from reasoning_profiles import (
            build_profiles
        )

        print(
            "\n=== Computing reasoning profiles ==="
        )

        build_profiles(suffix=args.suffix)



    if args.stage in [
        "entropy",
        "all",
    ]:

        from feature_analysis import (
            compute_entropy
        )

        print(
            "\n=== Computing reasoning concentration ==="
        )

        compute_entropy(suffix=args.suffix)



    if args.stage in [
        "distribution",
        "all",
    ]:

        from importance_distribution import (
            build_distribution
        )

        print(
            "\n=== Computing importance distributions ==="
        )

        build_distribution(suffix=args.suffix)



if __name__ == "__main__":
    main()