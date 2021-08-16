#include <imago/imago.h>
#include <SFML/Graphics.hpp>

float randomFloat(float min, float max) {
    float random = ((float)rand()) / (float)RAND_MAX;

    // generate (in your case) a float between 0 and (4.5-.78)
    // then add .78, giving you a float between .78 and 4.5
    float range = max - min;
    return (random * range) + min;
}

int main(int argc, char **argv) {
    int neuronsCount;
    int synapsesDensity;

    // Input handling.
    switch (argc) {
        case 1:
            neuronsCount = 100;
            synapsesDensity = 2;
            break;
        case 2:
            neuronsCount = atoi(argv[1]);
            synapsesDensity = 2;
            break;
        case 3:
            neuronsCount = atoi(argv[1]);
            synapsesDensity = atoi(argv[2]);
            break;
        default:
            printf("Too many args.\n");
            break;
    }

    sf::VideoMode desktopMode = sf::VideoMode::getDesktopMode();


    srand(time(0));

    // Create network model.
    corticolumn column;
    dccol_init(&column, neuronsCount, synapsesDensity);

    float* xNeuronPositions = (float*) malloc(column.neurons_count * sizeof(float));
    float* yNeuronPositions = (float*) malloc(column.neurons_count * sizeof(float));

    for (int i = 0; i < column.neurons_count; i++) {
        xNeuronPositions[i] = randomFloat(0, 1);
        yNeuronPositions[i] = randomFloat(0, 1);
    }
    
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;

    // create the window
    sf::RenderWindow window(desktopMode, "Imago", sf::Style::Fullscreen, settings);

    // Run the program as long as the window is open
    while (window.isOpen()) {
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event)) {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        // Clear the window with black color
        window.clear(sf::Color::Black);

        // Draw neurons.
        for (int i = 0; i < column.neurons_count; i++) {
            sf::CircleShape neuronSpot;
            neuronSpot.setRadius(5.0f);
            neuronSpot.setFillColor(sf::Color::Blue);
            neuronSpot.setPosition(xNeuronPositions[i] * desktopMode.width, yNeuronPositions[i] * desktopMode.height);

            // Center the spot.
            neuronSpot.setOrigin(5.0f, 5.0f);
            window.draw(neuronSpot);
        }

        // Draw synapses.
        for (int i = 0; i < column.synapses_count; i++) {
            sf::Vertex line[] = {
                sf::Vertex({xNeuronPositions[column.synapses[i].input_neuron] * desktopMode.width, yNeuronPositions[column.synapses[i].input_neuron] * desktopMode.height}, sf::Color(255, 255, 31, 127)),
                sf::Vertex({xNeuronPositions[column.synapses[i].output_neuron] * desktopMode.width, yNeuronPositions[column.synapses[i].output_neuron] * desktopMode.height}, sf::Color(31, 127, 255, 127))
            };

            window.draw(line, 2, sf::Lines);
        }

        // end the current frame
        window.display();
    }

    return 0;
}