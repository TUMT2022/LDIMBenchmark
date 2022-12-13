from ldimbenchmark.generator.poulakis_network import generatePoulakisNetwork
import pytest


@pytest.mark.parametrize("number_of_pipes", range(1, 30))
def test_generate_with_number_of_pipes(number_of_pipes):
    assert (
        generatePoulakisNetwork(6, max_pipes=number_of_pipes).describe()["Links"]
        == number_of_pipes
    )


@pytest.mark.parametrize("number_of_nodes", range(3, 30))
def test_generate_with_number_of_junction(number_of_nodes):
    assert (
        generatePoulakisNetwork(6, max_junctions=number_of_nodes).describe()["Nodes"]
        == number_of_nodes
    )
