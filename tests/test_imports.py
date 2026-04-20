import unittest


class ImportSmokeTest(unittest.TestCase):
    def test_import_core_modules(self):
        import ctpe.run_stage2_extended_suite  # noqa: F401
        import ctpe.stage3_heavy_common  # noqa: F401
        import ctpe.run_stage3_heavy_benchmark  # noqa: F401
        import ctpe.run_stage4_regime_refinement  # noqa: F401
        import ctpe.run_stage5_clean_regime_suite  # noqa: F401


if __name__ == "__main__":
    unittest.main()
