#ifndef LIBRARY_H
#define LIBRARY_H

#include <sstream>
#include <vector>
#include <cctype>
#include <list>
#include <map>
#include <set>
#include <stack>
#include <queue>

template<typename T>
struct Vector : std::vector<T>
{
    using My = Vector;

    bool contains(const T& x) const
    {
        return std::find(My::begin(), My::end(), x) != My::end();
    }

    template<typename Pred>
    bool contains(Pred p) const
    {
        return std::find_if(My::begin(), My::end(), p) != My::end();
    }
};

template<typename T>
struct List : std::list<T>
{
    using My = List;

    bool contains(const T& x) const 
    {
        return My::find(x) != My::end();
    }

    template<typename Pred>
    bool contains(Pred p) const
    {
        return std::find_if(My::begin(), My::end(), p) != My::end();
    }
};

template<typename K, typename T>
struct Map : std::map<K, T>
{
    using My = Map;

    bool contains(const K& key) const
    {
        return My::find(key) != My::end();
    }

    template<typename Pred>
    bool contains(Pred p) const
    {
        return std::find_if(My::begin(), My::end(), p) != My::end();
    }
};

template<typename K>
struct Set : std::set<K>
{
    using My = Set;

    bool contains(const K& key) const
    {
        return My::find(key) != My::end();
    }

    template<typename Pred>
    bool contains(Pred p) const
    {
        return std::find_if(My::begin(), My::end(), p) != My::end();
    }
};

template<typename T>
struct Stack : std::stack<T>
{
    using My = Stack;

    bool contains(const T& x) const
    {
        return std::find(My::begin(), My::end(), x) != My::end();
    }

    template<typename Pred>
    bool contains(Pred p) const
    {
        return std::find_if(My::begin(), My::end(), p) != My::end();
    }
};

template<typename T>
struct Queue : std::queue<T>
{
    using My = Queue;

    bool contains(const T& x) const
    {
        return std::find(My::begin(), My::end(), x) != My::end();
    }

    template<typename Pred>
    bool contains(Pred p) const
    {
        return std::find_if(My::begin(), My::end(), p) != My::end();
    }
};

///////////////////////////// FREE FUNCTIONS /////////////////////////////

inline std::string peekNextLine(std::stringstream& ss)
{
    std::string nextLine = "";
    const std::stringstream::pos_type pos = ss.tellg();
    if (ss.peek() != EOF)
        std::getline(ss, nextLine);
    ss.seekg(pos);
    return nextLine;
}

inline std::string stripWhitespace(std::string s)
{
    s.erase(std::remove_if(s.begin(), s.end(), [](char c) { return std::isspace(c); }), s.end());
}

#endif // LIBRARY_H

