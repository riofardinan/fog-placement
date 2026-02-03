from yafs.population import Population
from yafs.distribution import exponentialDistribution


class JSONPopulation(Population):
    """
    Population that deploys sources from a JSON structure (usersDefinition format).
    Uses exponential distribution for request inter-arrival times.
    """

    def __init__(self, name, json_data, iteration=1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.data = json_data
        self.iteration = iteration

    def initial_allocation(self, sim, app_name):
        """Deploy sources for this app from self.data['sources']."""
        for item in self.data.get("sources", []):
            if item["app"] != app_name:
                continue
            idtopo = item["id_resource"]
            lambd = item["lambda"]
            app = sim.apps[app_name]
            msg = app.get_message(item["message"])

            d_dist = exponentialDistribution(
                name="Exp", lambd=lambd, seed=self.iteration
            )
            sim.deploy_source(
                app_name, id_node=idtopo, msg=msg, distribution=d_dist
            )
