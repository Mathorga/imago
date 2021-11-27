#include <imago/imago.h>
#include <SFML/Graphics.hpp>
#include <time.h>
#include <unistd.h>
#include <ncurses.h>

float randomFloat(float min, float max) {
    float random = ((float)rand()) / (float)RAND_MAX;

    float range = max - min;
    return (random * range) + min;
}

int main(int argc, char **argv) {
    int neuronsCount = 100;
    int synapsesDensity = 10;

    // Input handling.
    switch (argc) {
        case 2:
            neuronsCount = atoi(argv[1]);
            break;
        case 3:
            neuronsCount = atoi(argv[1]);
            synapsesDensity = atoi(argv[2]);
            break;
        default:
            break;
    }

    srand(time(0));

    // Create network model.
    braph column;
    dbraph_init(&column, neuronsCount, synapsesDensity);

    bool feed = false;
    bool running = true;

    char c;

    initscr();
    cbreak();
    noecho();
    scrollok(stdscr, TRUE);
    nodelay(stdscr, TRUE);

    // Run the program as long as the window is open.
    while (running) {
        usleep(1000);

        // User input character.
        c = getch();

        // clear();
        printw("Running\n");

        switch(c) {
            case 'q':
                running = false;
                break;
            case 'b':
                feed = !feed;
                break;
        }

        // Feed the column and tick it.
        neurons_count_t inputNeuronsCount = 4;
        neurons_count_t startingInputIndex = 0;
        if (feed && randomFloat(0, 1) < 0.4f) {
            braph_feed(&column, startingInputIndex, inputNeuronsCount, SYNAPSE_DEFAULT_VALUE);
        }
        braph_tick(&column);

        // Print spikes count.
        printw("Spikes count: %d\nFeeding: %s\n\n", column.spikes_count, (feed ? "true" : "false"));
    }

    endwin();

    return 0;
}
