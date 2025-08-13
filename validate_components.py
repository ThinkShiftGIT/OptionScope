"""
Validation script to check that all key components are working correctly.
This script performs basic checks of core functionality without needing
the full test infrastructure.
"""

import sys
import os
import importlib
from pprint import pprint

# Basic validation results
results = {
    "imports": {},
    "functionality": {}
}

def validate_import(module_name):
    """Validate that a module can be imported."""
    try:
        module = importlib.import_module(module_name)
        results["imports"][module_name] = "✅ Success"
        return module
    except Exception as e:
        results["imports"][module_name] = f"❌ Failed: {str(e)}"
        return None

def validate_functionality(name, function, *args, **kwargs):
    """Validate that a function executes without errors."""
    try:
        result = function(*args, **kwargs)
        results["functionality"][name] = "✅ Success"
        return result
    except Exception as e:
        results["functionality"][name] = f"❌ Failed: {str(e)}"
        return None

def main():
    """Run validation tests on key components."""
    print("Starting validation of OptionScope components...\n")
    
    # Ensure app module is in path
    sys.path.insert(0, os.path.abspath('.'))
    
    # Test imports of key modules
    print("Testing imports...")
    
    # Core modules
    data_providers = validate_import("app.core.data_providers")
    indicators = validate_import("app.core.indicators")
    strategies = validate_import("app.core.strategies.base")
    risk = validate_import("app.core.risk")
    scoring = validate_import("app.core.scoring")
    
    # Utility modules
    config = validate_import("app.utils.config")
    
    # UI modules
    components = validate_import("app.ui.components")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    
    # Test config functionality
    if config:
        test_config = {
            'test': {
                'nested': {
                    'value': 42
                }
            }
        }
        
        # Mock the get_config function for testing
        def mock_get_config():
            return test_config
        
        config.get_config = mock_get_config
        
        validate_functionality(
            "config.get_config_value",
            config.get_config_value,
            "test.nested.value"
        )
    
    # Test risk calculations
    if risk and strategies:
        # Create a dummy strategy candidate
        StrategyCandidate = strategies.StrategyCandidate
        OptionLeg = strategies.OptionLeg
        TradeAction = strategies.TradeAction
        OptionType = strategies.OptionType
        
        dummy_candidate = StrategyCandidate(
            strategy_name="Test Strategy",
            symbol="AAPL",
            dte_short=30,
            max_loss=1000.0,
            max_profit=500.0,
            buying_power_effect=2000.0,
            probability_of_profit=0.6,
            expected_return=5.0,
            legs=[
                OptionLeg(
                    action=TradeAction.BUY,
                    option_type=OptionType.CALL,
                    strike=155.0,
                    expiration_date="2023-06-15",
                    quantity=1,
                    price=3.0,
                    delta=0.4
                )
            ],
            delta=0.1,
            gamma=0.005,
            theta=-0.02,
            vega=0.05,
            avg_spread_pct=0.03,
            avg_open_interest=500,
            composite_score=75.0,
            estimated_price=1.5,
            notes="Test strategy candidate"
        )
        
        validate_functionality(
            "risk.calculate_max_loss",
            risk.calculate_max_loss,
            dummy_candidate
        )
        
        validate_functionality(
            "risk.calculate_liquidity_penalty",
            risk.calculate_liquidity_penalty,
            dummy_candidate
        )
    
    # Test scoring functionality
    if scoring and risk and strategies:
        candidates = [dummy_candidate]
        
        validate_functionality(
            "scoring.rank_candidates",
            scoring.rank_candidates,
            candidates,
            10000
        )
    
    # Print validation results
    print("\n=== VALIDATION RESULTS ===")
    print("\nImport Tests:")
    for module, result in results["imports"].items():
        print(f"{module}: {result}")
    
    print("\nFunctionality Tests:")
    for function, result in results["functionality"].items():
        print(f"{function}: {result}")
    
    # Count successful vs failed tests
    successful_imports = sum(1 for result in results["imports"].values() if result.startswith("✅"))
    successful_functions = sum(1 for result in results["functionality"].values() if result.startswith("✅"))
    total_imports = len(results["imports"])
    total_functions = len(results["functionality"])
    
    print(f"\nImports: {successful_imports}/{total_imports} successful")
    print(f"Functions: {successful_functions}/{total_functions} successful")
    
    if successful_imports == total_imports and successful_functions == total_functions:
        print("\n✅ All components validated successfully!")
    else:
        print("\n⚠️ Some components failed validation. See results above.")

if __name__ == "__main__":
    main()
