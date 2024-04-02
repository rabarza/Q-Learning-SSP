# Q-Learning-SSP

This project implements the Q-Learning algorithm to solve the SSP (Shortest Path Problem) using graphs of cities and graphs with a structure similar to a perceptron graph.

## Description

The Q-Learning algorithm is a reinforcement learning technique that aims to find the optimal policy for an agent in a given environment. In this project, we apply Q-Learning to solve the SSP, which involves finding the shortest path between two nodes in a graph.

We have two types of graphs in this project:

1. Graphs of Cities: These graphs represent a network of cities, where each location in the city is a node and the connections between locations are edges. The goal is to find the shortest path between two nodes.

2. Perceptron-like Graphs: These graphs have a structure similar to a perceptron, with input nodes, hidden nodes, and output nodes. The goal is to find the shortest path from the input nodes to the output nodes.

## Execution

To execute this project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory: `C:/Users/user/project-directory/q-learning_app`.
3. Install the required dependencies by running the following command:

   ```
   pip install -r requirements.txt
   ```

4. Run the main script by executing the following command:

   ```
   streamlit run app.py
   ```

   This will start the Q-Learning SSP Application were you can solve the SSP using specified graphs.

5. View the results. The shortest path and other relevant information will be displayed in the console output.

## Contributing

If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository and clone it to your local machine.
2. Create a new branch for your feature or bug fix.
3. Make your changes and test them thoroughly.
4. Commit your changes and push them to your forked repository.
5. Submit a pull request, explaining the changes you have made and why they should be merged.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
