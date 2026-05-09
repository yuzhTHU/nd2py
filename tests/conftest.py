import pytest

def pytest_addoption(parser):
    # 添加命令行参数
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="只运行慢速的集成测试"
    )
    parser.addoption(
        "--run-all", action="store_true", default=False, help="运行所有测试 (包含快速和慢速)"
    )
    parser.addoption(
        "--run-paid", action="store_true", default=False, help="运行收费的 API 测试"
    )

def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--run-slow")
    run_all = config.getoption("--run-all")
    run_paid = config.getoption("--run-paid")

    # 模式 4: 全量运行
    if run_all:
        return

    skip_fast = pytest.mark.skip(reason="已指定 --run-slow，跳过快速单元测试")
    skip_slow = pytest.mark.skip(reason="默认跳过慢速测试，如需执行请添加 --run-all 或 --run-slow")
    skip_paid = pytest.mark.skip(reason="默认跳过收费测试，如需执行请添加 --run-all 或 --run-paid")

    for item in items:
        # 如果测试被标记为 @pytest.mark.slow
        if "slow" in item.keywords:
            if not run_slow:
                # 模式 1: 默认模式下，跳过慢速测试
                item.add_marker(skip_slow)
        else:
            # 如果测试是普通的快速测试
            if run_slow:
                # 模式 2: 指定了只跑慢速时，跳过快速测试
                item.add_marker(skip_fast)

        # 如果测试被标记为 @pytest.mark.paid
        if "paid" in item.keywords:
            if not run_paid:
                item.add_marker(skip_paid)