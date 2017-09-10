#include "SpecificUtilities.h"

void display(const std::string& msg)
{
    std::cout << std::endl << msg << std::endl << std::endl;
}

void display(const std::string& label, const std::vector<double>& v)
{
    std::cout << label << " ";

    for (const auto& elem : v)
        std::cout << elem << " ";

    std::cout << std::endl;
}

void displayStats(const double error, const double epoch)
{
    // Report how well the training is working, average over recent samples
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "          Pass: " << epoch            << std::endl;
    std::cout << " Average error: " << error           << std::endl;
    std::cout << "-----------------------------------" << std::endl;
}