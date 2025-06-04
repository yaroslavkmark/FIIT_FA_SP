//
// Created by Des Caldnd on 2/28/2025.
//

#ifndef B_TREE_DISK_HPP
#define B_TREE_DISK_HPP

#include <iterator>
#include <utility>
#include <vector>
#include <concepts>
#include <iostream>
#include <stack>
#include <fstream>
#include <optional>
#include <cstddef>
#include <filesystem>

#pragma pack(push, 1)
#pragma pack(pop)

template<typename compare, typename tkey>
concept compator = requires(const compare c, const tkey &lhs, const tkey &rhs)
                   {
                       { c(lhs, rhs) } -> std::same_as<bool>;
                   } && std::copyable<compare> && std::default_initializable<compare>;

template<typename f_iter, typename tkey, typename tval>
concept input_iterator_for_pair = std::input_iterator<f_iter> && std::same_as<typename std::iterator_traits<
        f_iter>::value_type, std::pair<tkey, tval> >;

template<typename T>
concept serializable = requires(const T t, std::fstream &s)
                       {
                           { t.serialize(s) };
                           { T::deserialize(s) } -> std::same_as<T>;
                           { t.serialize_size() } -> std::same_as<size_t>;
                       } && std::copyable<T>;

struct SerializableInt {
    int data;

    void serialize(std::fstream &stream) const {
        stream.write(reinterpret_cast<const char *>(&data), sizeof(int));
    }

    static SerializableInt deserialize(std::fstream &stream) {
        SerializableInt obj;
        stream.read(reinterpret_cast<char *>(&obj.data), sizeof(int));
        return obj;
    }

    size_t serialize_size() const {
        return sizeof(int);
    }

    operator int() const { return data; }
};


struct SerializableString {
    std::string data;

    SerializableString() = default;

    explicit SerializableString(const std::string &s) : data(s) {
    }

    explicit SerializableString(std::string &&s) : data(std::move(s)) {
    }

    explicit SerializableString(const char *s) : data(s) {
    }

    SerializableString &operator=(const char *s) {
        data = s;
        return *this;
    }

    // Сериализация в бинарный поток
    void serialize(std::fstream &stream) const {
        size_t size = data.size();
        stream.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
        if (size > 0)
            stream.write(data.data(), size);
    }

    std::string &get() noexcept { return data; }

    // Десериализация из бинарного потока
    static SerializableString deserialize(std::fstream &stream) {
        size_t size;
        stream.read(reinterpret_cast<char *>(&size), sizeof(size_t));
        std::string str(size, '\0');
        if (size > 0)
            stream.read(&str[0], size);
        return SerializableString(std::move(str));
    }

    // Размер сериализованной строки
    size_t serialize_size() const {
        return sizeof(size_t) + data.size();
    }

    // Операторы сравнения для удобства использования в тестах и B-дереве
    bool operator==(const SerializableString &other) const noexcept {
        return data == other.data;
    }

    bool operator>=(const SerializableString &other) const noexcept {
        return data >= other.data;
    }

    bool operator<=(const SerializableString &other) const noexcept {
        return data <= other.data;
    }

    bool operator<(const SerializableString &other) const noexcept {
        return data < other.data;
    }
    bool operator>(const SerializableString &other) const noexcept {
        return data > other.data;
    }

    bool operator!=(const SerializableString &other) const noexcept {
        return data != other.data;
    }

    // Неявное преобразование к std::string
    operator const std::string &() const noexcept { return data; }
    operator std::string &() noexcept { return data; }
};

template<serializable T>
struct SerializableVector {
    std::vector<T> data;

    SerializableVector() = default;

    SerializableVector(const std::vector<T> &v) : data(v) {
    }

    void serialize(std::fstream &stream) const {
        size_t sz = data.size();
        stream.write(reinterpret_cast<const char *>(&sz), sizeof(size_t));
        for (const auto &elem: data) elem.serialize(stream);
    }

    static SerializableVector<T> deserialize(std::fstream &stream) {
        size_t sz;
        stream.read(reinterpret_cast<char *>(&sz), sizeof(size_t));
        std::vector<T> v(sz);
        for (size_t i = 0; i < sz; ++i) v[i] = T::deserialize(stream);
        return SerializableVector<T>(v);
    }

    size_t serialize_size() const {
        size_t total = sizeof(size_t);
        for (const auto &elem: data) total += elem.serialize_size();
        return total;
    }

    operator std::vector<T>() const { return data; }
};

inline void write_node_header(std::fstream &stream, size_t size, bool is_leaf, size_t pos) {
    stream.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
    uint8_t leaf = is_leaf ? 1 : 0;
    stream.write(reinterpret_cast<const char *>(&leaf), sizeof(uint8_t));
    stream.write(reinterpret_cast<const char *>(&pos), sizeof(size_t));
}

inline void read_node_header(std::fstream &stream, size_t &size, bool &is_leaf, size_t &pos) {
    stream.read(reinterpret_cast<char *>(&size), sizeof(size_t));
    uint8_t leaf;
    stream.read(reinterpret_cast<char *>(&leaf), sizeof(uint8_t));
    is_leaf = (leaf != 0);
    stream.read(reinterpret_cast<char *>(&pos), sizeof(size_t));
}

template<serializable tkey, serializable tvalue, compator<tkey> compare = std::less<tkey>, std::size_t t = 2>
class B_tree_disk final : private compare {
public:
    using tree_data_type = std::pair<tkey, tvalue>;
    using tree_data_type_const = std::pair<tkey, tvalue>;

private:
    static constexpr const size_t minimum_keys_in_node = t - 1;
    static constexpr const size_t maximum_keys_in_node = 2 * t - 1;

    // region comparators declaration

    inline bool compare_keys(const tkey &lhs, const tkey &rhs) const;

    inline bool compare_pairs(const tree_data_type &lhs, const tree_data_type &rhs) const;

    // endregion comparators declaration

public:
    struct btree_disk_node {
        size_t size; // кол-во заполненных ячеек
        bool _is_leaf;
        size_t position_in_disk;
        std::vector<tree_data_type> keys;
        std::vector<size_t> pointers;

        void serialize(std::fstream &stream, std::fstream &stream_for_data) const;

        static btree_disk_node deserialize(std::fstream &stream, std::fstream &stream_for_data);

        explicit btree_disk_node(bool is_leaf);

        btree_disk_node();
    };

private:
    friend btree_disk_node;

    //logger* _logger;

    std::fstream _file_for_tree;

    std::fstream _file_for_key_value;

    //    btree_disk_node _root;

public:
    size_t _position_root; //

private:
    btree_disk_node _current_node;

    //logger* get_logger() const noexcept override;


public:
    size_t _count_of_node; //только растет

    // region constructors declaration

    explicit B_tree_disk(const std::string &file_path, const compare &cmp = compare(), void *logger = nullptr);


    // endregion constructors declaration

    // region five declaration

    B_tree_disk(B_tree_disk &&other) noexcept = default;

    B_tree_disk &operator=(B_tree_disk &&other) noexcept = default;

    B_tree_disk(const B_tree_disk &other) = delete;

    B_tree_disk &operator=(const B_tree_disk &other) = delete;

    ~B_tree_disk() noexcept = default;

    // endregion five declaration

    // region iterators declaration

    class btree_disk_const_iterator {
        std::stack<std::pair<size_t, size_t> > _path;
        size_t _index;
        B_tree_disk<tkey, tvalue, compare, t> *_tree;

    public:
        using value_type = tree_data_type_const;
        using reference = value_type &;
        using pointer = value_type *;
        using iterator_category = std::forward_iterator_tag;
        using difference_type = ptrdiff_t;

        using self = btree_disk_const_iterator;

        friend class B_tree_disk;

        value_type operator*() noexcept;

        self &operator++();

        self operator++(int);

        self &operator--();

        self operator--(int);

        bool operator==(self &other) noexcept;

        bool operator!=(self &other) noexcept;

        explicit btree_disk_const_iterator(B_tree_disk<tkey, tvalue, compare, t> *tree,
                                           const std::stack<std::pair<size_t, size_t> > &path = std::stack<std::pair<
                                                   size_t, size_t> >(), size_t index = 0);
    };

    friend class btree_disk_const_iterator;

    std::optional<tvalue> at(const tkey &); //либо пустое, либо tvalue//std::nullopt

    btree_disk_const_iterator begin();

    btree_disk_const_iterator end();

    //std::vector<tree_data_type_const> find_range(const tkey& lower, const tkey& upper) const;

    std::pair<btree_disk_const_iterator, btree_disk_const_iterator> find_range(
            const tkey &lower, const tkey &upper, bool include_lower = true, bool include_upper = false);

    /*
     * Does nothing if key exists
     * Second return value is true, when inserted
     */
    bool insert(const tree_data_type &data);

    /*
     * Updates value if key exists
     */
    bool update(const tree_data_type &data);

    /*
     * Return true if deleted
     */
    bool erase(const tkey &key);

    bool is_valid() const noexcept;


    std::pair<std::stack<std::pair<size_t, size_t> >, std::pair<size_t, bool> > find_path(const tkey &key);

public:
    btree_disk_node disk_read(size_t position);


    void check_tree(size_t pos, size_t depth);

    void disk_write(btree_disk_node &node);
    void print(std::ostream &os = std::cout) const noexcept {
        std::cout << _position_root << "\n";
        if (_count_of_node == 0) {
            os << "<empty>\n";
            return;
        }
        print_node(os, _position_root, 0);
    }
private:
    std::pair<size_t, bool> find_index(const tkey &key, btree_disk_node &node) const noexcept;

    void insert_array(btree_disk_node &node, size_t right_node, const tree_data_type &data, size_t index) noexcept;

    void split_node(std::stack<std::pair<size_t, size_t> > &path);

    btree_disk_node remove_array(btree_disk_node &node, size_t index, bool remove_left_ptr = true) noexcept;

    void rebalance_node(std::stack<std::pair<size_t, size_t> > &path, btree_disk_node &node, size_t &index);

    void print_root_position() noexcept;

    void print_node(std::ostream &os, std::size_t pos, int level) const {
        auto node = const_cast<B_tree_disk*>(this)->disk_read(pos);;
        // отступ level×4 пробелов
        os << std::string(level * 4, ' ')
           << (node._is_leaf ? "Leaf" : "Internal")
           << " (pos=" << pos << ") keys:";
        for (auto &kv : node.keys) {
            // печатаем пару (ключ:значение)
            os << " (" << kv.first.data << ":" << kv.second.data << ")";
        }
        os << "\n";
        // если не лист — рекурсивно печатаем детей
        if (!node._is_leaf) {
            for (std::size_t i = 0; i <= node.keys.size(); ++i) {
                print_node(os, node.pointers[i], level + 1);
            }
        }
    }
};

template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
bool B_tree_disk<tkey, tvalue, compare, t>::is_valid() const noexcept {

}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
bool B_tree_disk<tkey, tvalue, compare, t>::erase(const tkey &key) {
    // Проверка на пустое дерево
    if (_position_root == static_cast<size_t>(-1)) {
        return false;
    }

    // Находим путь до узла, содержащего ключ
    auto [path, result] = find_path(key);
    auto [index, found] = result;

    // Ключ не найден - возвращаем false
    if (!found) {
        return false;
    }

    // Получаем текущий узел, содержащий ключ
    size_t current_pos = path.top().first;
    path.pop();
    btree_disk_node current = disk_read(current_pos);

    // СЛУЧАЙ 1: Ключ находится в листе
    if (current._is_leaf) {
        // Удаляем ключ из листа
        current.keys.erase(current.keys.begin() + index);
        current.size--;
        disk_write(current);

        // Проверяем необходимость ребалансировки
        if (current.size < minimum_keys_in_node && current_pos != _position_root) {
            rebalance_node(path, current, index);
        } else if (current_pos == _position_root && current.size == 0) {
            // Если удалили последний ключ из корня, дерево становится пустым
            _position_root = static_cast<size_t>(-1);
            print_root_position();
        }
        return true;
    }

    // СЛУЧАЙ 2: Ключ находится во внутреннем узле

    // Проверяем левое поддерево
    size_t left_child_pos = current.pointers[index];
    btree_disk_node left_child = disk_read(left_child_pos);

    // Случай 2.1: Левый потомок имеет достаточно ключей
    if (left_child.size > minimum_keys_in_node) {
        // Находим предшественника (самый правый ключ в левом поддереве)
        btree_disk_node pred_node = left_child;
        size_t pred_pos = left_child_pos;

        std::stack<std::pair<size_t, size_t>> pred_path;
        pred_path.push({current_pos, index});

        // Спускаемся вниз до самого правого листа
        while (!pred_node._is_leaf) {
            pred_path.push({pred_pos, pred_node.size});
            pred_pos = pred_node.pointers[pred_node.size];
            pred_node = disk_read(pred_pos);
        }

        // Заменяем удаляемый ключ предшественником
        current.keys[index] = pred_node.keys[pred_node.size - 1];
        disk_write(current);

        // Удаляем предшественник из листа
        pred_node.keys.erase(pred_node.keys.begin() + pred_node.size - 1);
        pred_node.size--;
        disk_write(pred_node);

        // Ребалансировка, если необходимо
        if (pred_node.size < minimum_keys_in_node && pred_pos != _position_root) {
            size_t pred_idx = pred_node.size;
            rebalance_node(pred_path, pred_node, pred_idx);
        }
        return true;
    }

    // Проверяем правое поддерево
    size_t right_child_pos = current.pointers[index + 1];
    btree_disk_node right_child = disk_read(right_child_pos);

    // Случай 2.2: Правый потомок имеет достаточно ключей
    if (right_child.size > minimum_keys_in_node) {
        // Находим преемника (самый левый ключ в правом поддереве)
        btree_disk_node succ_node = right_child;
        size_t succ_pos = right_child_pos;

        std::stack<std::pair<size_t, size_t>> succ_path;
        succ_path.push({current_pos, index + 1});

        // Спускаемся вниз до самого левого листа
        while (!succ_node._is_leaf) {
            succ_path.push({succ_pos, 0});
            succ_pos = succ_node.pointers[0];
            succ_node = disk_read(succ_pos);
        }

        // Заменяем удаляемый ключ преемником
        current.keys[index] = succ_node.keys[0];
        disk_write(current);

        // Удаляем преемник из листа
        succ_node.keys.erase(succ_node.keys.begin());
        succ_node.size--;
        disk_write(succ_node);

        // Ребалансировка, если необходимо
        if (succ_node.size < minimum_keys_in_node && succ_pos != _position_root) {
            size_t succ_idx = 0;
            rebalance_node(succ_path, succ_node, succ_idx);
        }
        return true;
    }

    // Случай 2.3: Оба потомка имеют минимальное количество ключей

    // Объединяем левого и правого потомков с ключом из текущего узла

    // Добавляем все ключи из правого потомка в левый
    for (size_t i = 0; i < right_child.size; i++) {
        left_child.keys.push_back(right_child.keys[i]);
        left_child.size++;
    }

    // Если не листовые узлы, переносим указатели
    if (!left_child._is_leaf) {
        for (size_t i = 0; i <= right_child.size; i++) {
            left_child.pointers.push_back(right_child.pointers[i]);
        }
    }

    // Записываем объединенный узел
    disk_write(left_child);

    // Удаляем ключ из текущего узла и обновляем указатели
    current.keys.erase(current.keys.begin() + index);
    current.pointers.erase(current.pointers.begin() + (index + 1));
    current.size--;
    disk_write(current);

    // Если корень стал пустым, обновляем корень
    if (current_pos == _position_root && current.size == 0) {
        _position_root = left_child_pos;
        print_root_position();
    }
        // Если узел требует ребалансировки
    else if (current.size < minimum_keys_in_node && current_pos != _position_root) {
        rebalance_node(path, current, index);
    }
    return true;
}



template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
void B_tree_disk<tkey, tvalue, compare, t>::print_root_position() noexcept {
    if (!_file_for_tree.is_open())
        return;


    _file_for_tree.seekp(sizeof(size_t), std::ios::beg);
    _file_for_tree.write(reinterpret_cast<const char *>(&_position_root), sizeof(size_t));
    _file_for_tree.flush();
}

template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
void B_tree_disk<tkey, tvalue, compare, t>::rebalance_node(
        std::stack<std::pair<size_t, size_t>> &path,
        btree_disk_node &node,
        size_t &index) {

    // Если путь пуст или узел имеет достаточно ключей, выходим
    if (path.empty() || node.size >= minimum_keys_in_node) {
        return;
    }

    // Получаем родительский узел
    size_t parent_pos = path.top().first;
    size_t parent_idx = path.top().second;
    path.pop();

    btree_disk_node parent = disk_read(parent_pos);

    // Определяем позицию текущего узла среди потомков родителя
    size_t node_idx = 0;
    while (node_idx <= parent.size && parent.pointers[node_idx] != node.position_in_disk) {
        node_idx++;
    }

    // Стратегия 1: Попытаться заимствовать ключ у левого соседа
    if (node_idx > 0) {
        size_t left_sibling_pos = parent.pointers[node_idx - 1];
        btree_disk_node left_sibling = disk_read(left_sibling_pos);

        // Если у левого соседа есть запасной ключ
        if (left_sibling.size > minimum_keys_in_node) {
            // Сдвигаем все ключи в текущем узле вправо
            node.keys.insert(node.keys.begin(), parent.keys[node_idx - 1]);
            node.size++;

            // Перемещаем последний ключ из левого соседа в родителя
            parent.keys[node_idx - 1] = left_sibling.keys[left_sibling.size - 1];

            // Если не лист, перемещаем соответствующий указатель
            if (!node._is_leaf) {
                node.pointers.insert(node.pointers.begin(), left_sibling.pointers[left_sibling.size]);
                left_sibling.pointers.pop_back();
            }

            // Удаляем последний ключ из левого соседа
            left_sibling.keys.pop_back();
            left_sibling.size--;

            // Записываем обновленные узлы
            disk_write(left_sibling);
            disk_write(parent);
            disk_write(node);
            return;
        }
    }

    // Стратегия 2: Попытаться заимствовать ключ у правого соседа
    if (node_idx < parent.size) {
        size_t right_sibling_pos = parent.pointers[node_idx + 1];
        btree_disk_node right_sibling = disk_read(right_sibling_pos);

        // Если у правого соседа есть запасной ключ
        if (right_sibling.size > minimum_keys_in_node) {
            // Добавляем ключ из родителя в текущий узел
            node.keys.push_back(parent.keys[node_idx]);
            node.size++;

            // Перемещаем первый ключ из правого соседа в родителя
            parent.keys[node_idx] = right_sibling.keys[0];

            // Если не лист, перемещаем соответствующий указатель
            if (!node._is_leaf) {
                node.pointers.push_back(right_sibling.pointers[0]);
                right_sibling.pointers.erase(right_sibling.pointers.begin());
            }

            // Удаляем первый ключ из правого соседа
            right_sibling.keys.erase(right_sibling.keys.begin());
            right_sibling.size--;

            // Записываем обновленные узлы
            disk_write(right_sibling);
            disk_write(parent);
            disk_write(node);
            return;
        }
    }

    // Стратегия 3: Объединение с соседом

    // Вариант 3.1: Объединение с левым соседом
    if (node_idx > 0) {
        size_t left_sibling_pos = parent.pointers[node_idx - 1];
        btree_disk_node left_sibling = disk_read(left_sibling_pos);

        // Добавляем разделяющий ключ из родителя в левого соседа
        left_sibling.keys.push_back(parent.keys[node_idx - 1]);
        left_sibling.size++;

        // Добавляем все ключи из текущего узла в левого соседа
        for (size_t i = 0; i < node.size; i++) {
            left_sibling.keys.push_back(node.keys[i]);
            left_sibling.size++;
        }

        // Если не листовые узлы, переносим указатели
        if (!node._is_leaf) {
            for (size_t i = 0; i <= node.size; i++) {
                left_sibling.pointers.push_back(node.pointers[i]);
            }
        }

        // Записываем обновленного левого соседа
        disk_write(left_sibling);

        // Удаляем ключ-разделитель из родителя и указатель на текущий узел
        parent.keys.erase(parent.keys.begin() + (node_idx - 1));
        parent.pointers.erase(parent.pointers.begin() + node_idx);
        parent.size--;
        disk_write(parent);

        // Если родитель стал корнем с 0 ключей, обновляем корень
        if (parent_pos == _position_root && parent.size == 0) {
            _position_root = left_sibling_pos;
            print_root_position();
        }
            // Если родитель требует ребалансировки
        else if (parent.size < minimum_keys_in_node && parent_pos != _position_root) {
            rebalance_node(path, parent, parent_idx);
        }

        return;
    }

    // Вариант 3.2: Объединение с правым соседом
    if (node_idx <= parent.size) {
        size_t right_sibling_pos = parent.pointers[node_idx + 1];
        btree_disk_node right_sibling = disk_read(right_sibling_pos);

        // Добавляем разделяющий ключ из родителя в текущий узел
        node.keys.push_back(parent.keys[node_idx]);
        node.size++;

        // Добавляем все ключи из правого соседа в текущий узел
        for (size_t i = 0; i < right_sibling.size; i++) {
            node.keys.push_back(right_sibling.keys[i]);
            node.size++;
        }

        // Если не листовые узлы, переносим указатели
        if (!node._is_leaf) {
            for (size_t i = 0; i <= right_sibling.size; i++) {
                node.pointers.push_back(right_sibling.pointers[i]);
            }
        }

        // Записываем обновленный текущий узел
        disk_write(node);

        // Удаляем ключ-разделитель из родителя и указатель на правого соседа
        parent.keys.erase(parent.keys.begin() + node_idx);
        parent.pointers.erase(parent.pointers.begin() + (node_idx + 1));
        parent.size--;
        disk_write(parent);

        // Если родитель стал корнем с 0 ключей, обновляем корень
        if (parent_pos == _position_root && parent.size == 0) {
            _position_root = node.position_in_disk;
            print_root_position();
        }
            // Если родитель требует ребалансировки
        else if (parent.size < minimum_keys_in_node && parent_pos != _position_root) {
            rebalance_node(path, parent, parent_idx);
        }
    }
}

template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
typename B_tree_disk<tkey, tvalue, compare, t>::btree_disk_node
B_tree_disk<tkey, tvalue, compare, t>::remove_array(
        btree_disk_node &node,
        size_t index,
        bool remove_left_ptr) noexcept {
    if (index >= node.keys.size()) return node;

    // Удаление ключа и соответствующего указателя
    node.keys.erase(node.keys.begin() + index);
    if (!node._is_leaf) {
        if (remove_left_ptr) {
            node.pointers.erase(node.pointers.begin() + index);
        } else {
            node.pointers.erase(node.pointers.begin() + index + 1);
        }
    }
    node.size--;

    // Проверка на необходимость перебалансировки
    if (node.size >= minimum_keys_in_node || node.position_in_disk == _position_root) {
        disk_write(node);
        return node;
    }

    // Стек пути для восстановления баланса
    std::stack<std::pair<size_t, size_t> > path;
    path.push({node.position_in_disk, index});

    rebalance_node(path, node, index);
    return disk_read(node.position_in_disk);
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
bool B_tree_disk<tkey, tvalue, compare, t>::update(const tree_data_type &data) {
    auto [path, result] = find_path(data.first);
    auto [index, found] = result;

    if (!found)
        return false;

    size_t node_pos = path.top().first;
    btree_disk_node node = disk_read(node_pos);
    node.keys[index].second = data.second;

    disk_write(node);
    return true;
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
bool B_tree_disk<tkey, tvalue, compare, t>::insert(const B_tree_disk::tree_data_type &data) {
    // 1) Находим путь до листа и позицию вставки
    auto [path, result] = find_path(data.first);
    auto [index, found] = result;
    if (found) {
        // Ключ уже есть – ничего не делаем
        return false;
    }

    // 2) Вынимаем из стека позицию листа
    size_t current_position = path.top().first;
    path.pop();

    // 3) Читаем лист
    btree_disk_node current_node = disk_read(current_position);
    // std::cout << "before disk_write: size=" << current_node.size << ", keys: ";
    // for (auto &pair : current_node.keys) std::cout << pair.first.data << " ";
    // std::cout << std::endl;

    // 4) Вставляем новый элемент
    current_node.keys.insert(current_node.keys.begin() + index, data);
    ++current_node.size;

    // 5) Обновляем вектор указателей и сразу записываем узел на диск
    current_node.pointers.resize(maximum_keys_in_node + 2, 0);
    disk_write(current_node);
    // std::cout << "after disk_write:  size=" << current_node.size << ", keys: ";
    // for (auto &pair : current_node.keys) std::cout << pair.first.data << " ";
    // std::cout << std::endl;

    // 6) И только теперь проверяем на переполнение
    if (current_node.size > maximum_keys_in_node) {
        // Передаём в split_node путь с позицией и индексом вставленного ключа
        path.push({ current_position, index });
        split_node(path);
    }

    return true;
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
void B_tree_disk<tkey, tvalue, compare, t>::split_node(
        std::stack<std::pair<size_t, size_t>>& path
) {
    if (path.empty()) return;

    // 1. Извлекаем узел
    size_t current_pos = path.top().first;
    path.pop();
    btree_disk_node current = disk_read(current_pos);

    // защита от мусора
    current.pointers.resize(maximum_keys_in_node + 2, 0);

    // 2. Правый узел
    btree_disk_node right(current._is_leaf);
    right.position_in_disk = this->_count_of_node++;
    right.pointers.resize(maximum_keys_in_node + 2, 0);

    // 3. Считаем середину ПО ВЕКТОРУ ключей
    size_t mid = current.keys.size() / 2;
    tree_data_type middle_key = current.keys[mid];

    // 4. Переносим ключи > mid в правый узел
    right.keys.assign(
            current.keys.begin() + mid + 1,
            current.keys.end()
    );
    current.keys.erase(
            current.keys.begin() + mid,
            current.keys.end()
    );

    // 5. Перенос указателей (если не лист)
    if (!current._is_leaf) {
        right.pointers.assign(
                current.pointers.begin() + mid + 1,
                current.pointers.end()
        );
        current.pointers.erase(
                current.pointers.begin() + mid + 1,
                current.pointers.end()
        );
        right.pointers.resize(maximum_keys_in_node + 2, 0);
        current.pointers.resize(maximum_keys_in_node + 2, 0);
    }

    // 6. Обновляем размеры и пишем на диск
    current.size = current.keys.size();
    right.size   = right.keys.size();
    disk_write(current);
    disk_write(right);

    // 7. Вставляем в родителя или создаём новый корень
    if (path.empty()) {
        btree_disk_node root(false);
        root.position_in_disk = this->_count_of_node++;
        root.keys.push_back(middle_key);
        root.pointers[0] = current_pos;
        root.pointers[1] = right.position_in_disk;
        root.size = 1;
        root.pointers.resize(maximum_keys_in_node + 2, 0);
        _position_root = root.position_in_disk;
        disk_write(root);
        print_root_position();
        return;
    }

    auto [parent_pos, parent_idx] = path.top();
    btree_disk_node parent = disk_read(parent_pos);
    parent.keys.insert(parent.keys.begin() + parent_idx, middle_key);
    parent.pointers.insert(
            parent.pointers.begin() + parent_idx + 1,
            right.position_in_disk
    );
    ++parent.size;
    parent.pointers.resize(maximum_keys_in_node + 2, 0);
    disk_write(parent);
    if (parent.size > maximum_keys_in_node)
        split_node(path);
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
void B_tree_disk<tkey, tvalue, compare, t>::insert_array(btree_disk_node &node, size_t right_node,
                                                         const tree_data_type &data, size_t index) noexcept {
    node.keys.insert(node.keys.begin() + index, data);
    node.pointers.insert(node.pointers.begin() + index + 1, right_node);
    node.size++;

    disk_write(node);
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
std::pair<std::stack<std::pair<size_t, size_t> >, std::pair<size_t, bool> >
B_tree_disk<tkey, tvalue, compare, t>::find_path(const tkey &key) {
    std::stack<std::pair<size_t, size_t> > path;

    if (_position_root == static_cast<size_t>(-1)) {
        return {path, {0, false}};
    }

    size_t current_position = _position_root;

    btree_disk_node current_node = disk_read(current_position);

    // std::cout << "root: size=" << current_node.size
    //         << ", is_leaf=" << current_node._is_leaf
    //         << ", pointers={";
    // for (auto p: current_node.pointers) std::cout << p << " ";
    // std::cout << "}\n";

    while (true) {
        auto [index, key_found] = find_index(key, current_node);

        if (key_found) {
            path.push({current_position, index});
            return {path, {index, true}};
        }

        path.push({current_position, index});

        if (current_node._is_leaf) {
            return {path, {index, false}};
        }
        // std::cout << "cur pos: " << current_position << ", go to: " << current_node.pointers[index] << std::endl;
        current_position = current_node.pointers[index];
        current_node = disk_read(current_position);
    }
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
std::pair<size_t, bool> B_tree_disk<tkey, tvalue, compare, t>::find_index(
        const tkey &key, btree_disk_node &node) const noexcept {
    size_t i = 0;
    // std::cout << "find_index: key=" << key.data << ", keys: ";
    // for (auto &pair: node.keys) std::cout << pair.first.data << " ";
    // std::cout << std::endl;

    while (i < node.keys.size() && compare_keys(node.keys[i].first, key)) {
        // std::cout << "compare_keys(" << node.keys[i].first.data << ", " << key.data << ") = "
        // << compare_keys(node.keys[i].first, key) << std::endl;
        ++i;
    }

    bool found = (i < node.keys.size() && !compare_keys(key, node.keys[i].first));
    // std::cout << "index = " << i << ", found = " << found << std::endl;
    return {i, found};
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
void B_tree_disk<tkey, tvalue, compare, t>::btree_disk_node::serialize(
        std::fstream &tree_stream,
        std::fstream &data_stream) const {
    // пишем header
    write_node_header(tree_stream, size, _is_leaf, position_in_disk);

    // пишем указатели
    for (size_t i = 0; i < maximum_keys_in_node + 2; ++i) {
        size_t val = (i < pointers.size() ? pointers[i] : 0);
        tree_stream.write(reinterpret_cast<const char *>(&val), sizeof(size_t));
    }

    // пишем позиции + сам ключ/значение в data_stream
    for (size_t i = 0; i < maximum_keys_in_node + 1; ++i) {
        size_t pos = 0;
        if (i < keys.size()) {
            // идём в конец data-файла и запоминаем оффсет
            data_stream.seekp(0, std::ios::end);
            pos = static_cast<size_t>(data_stream.tellp());

            // сериализуем ключ и значение
            keys[i].first.serialize(data_stream);
            keys[i].second.serialize(data_stream);

            data_stream.flush();
        }
        // пишем в .tree файл позицию этого ключа (или 0)
        tree_stream.write(reinterpret_cast<const char *>(&pos), sizeof(size_t));
    }
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
void B_tree_disk<tkey, tvalue, compare, t>::disk_write(btree_disk_node &node) {
    const size_t header_size = 2 * sizeof(size_t); // count_of_node + position_root

    const size_t node_size =
            sizeof(size_t) + // size
            sizeof(uint8_t) + // is_leaf
            sizeof(size_t) + // position_in_disk
            (maximum_keys_in_node + 1) * sizeof(size_t) + // pointers
            (maximum_keys_in_node + 2) * sizeof(size_t); // keys positions

    _file_for_tree.seekp(header_size + node.position_in_disk * node_size, std::ios::beg);
    node.serialize(_file_for_tree, _file_for_key_value);
    _file_for_tree.flush();
    _file_for_key_value.flush();
}

template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
typename B_tree_disk<tkey, tvalue, compare, t>::btree_disk_node
B_tree_disk<tkey, tvalue, compare,
        t>::btree_disk_node::deserialize(std::fstream &stream, std::fstream &stream_for_data) {
    btree_disk_node node;

    // Чтение заголовка
    read_node_header(stream, node.size, node._is_leaf, node.position_in_disk);

    // Чтение указателей
    node.pointers.resize(maximum_keys_in_node + 2);
    for (auto &p: node.pointers)
        stream.read(reinterpret_cast<char *>(&p), sizeof(size_t));

    // Чтение ключей
    node.keys.clear();
    for (size_t i = 0; i < maximum_keys_in_node + 1; ++i) {
        size_t pos = 0;
        stream.read(reinterpret_cast<char *>(&pos), sizeof(size_t));
        if (node.keys.size() < node.size) {
            stream_for_data.seekg(pos);
            tree_data_type pair;
            pair.first = tkey::deserialize(stream_for_data);
            pair.second = tvalue::deserialize(stream_for_data);
            node.keys.push_back(pair);
        }
    }

    return node;
}

template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
typename B_tree_disk<tkey, tvalue, compare, t>::btree_disk_node
B_tree_disk<tkey, tvalue, compare, t>::disk_read(size_t node_position) {
    const size_t header_size = 2 * sizeof(size_t);
    const size_t node_size =
            sizeof(size_t) + // size
            sizeof(uint8_t) + // is_leaf
            sizeof(size_t) + // position_in_disk
            (maximum_keys_in_node + 2) * sizeof(size_t) + // pointers
            (maximum_keys_in_node + 1) * sizeof(size_t); // keys positions

    _file_for_tree.seekg(header_size + node_position * node_size, std::ios::beg);
    return btree_disk_node::deserialize(_file_for_tree, _file_for_key_value);
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
B_tree_disk<tkey, tvalue, compare, t>::btree_disk_node::btree_disk_node(bool is_leaf) : _is_leaf(is_leaf), size(0),
                                                                                        position_in_disk(0) {
    pointers.resize(maximum_keys_in_node + 2, 0);
}

template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
B_tree_disk<tkey, tvalue, compare,
        t>::btree_disk_node::btree_disk_node() : _is_leaf(true), size(0), position_in_disk(0) {
    pointers.resize(maximum_keys_in_node + 2, 0);
}

template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
bool B_tree_disk<tkey, tvalue, compare, t>::compare_pairs(const tree_data_type &lhs, const tree_data_type &rhs) const {
    return compare_keys(lhs.first, rhs.first);
}

template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
bool B_tree_disk<tkey, tvalue, compare, t>::compare_keys(const tkey &lhs, const tkey &rhs) const {
    return compare::operator()(lhs, rhs);
}

template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
B_tree_disk<tkey, tvalue, compare, t>::B_tree_disk(
        const std::string &file_path,
        const compare &cmp,
        void *logger)
        : compare(cmp) {
    std::string tree_file = file_path + ".tree";
    std::string data_file = file_path + ".data";

    bool files_exist =
            std::filesystem::exists(tree_file) &&
            std::filesystem::exists(data_file);

    if (!files_exist) {
        // Создаем и чистим оба файла
        _file_for_tree.open(tree_file,
                            std::ios::out | std::ios::binary | std::ios::trunc);
        _file_for_tree.close();

        _file_for_key_value.open(data_file,
                                 std::ios::out | std::ios::binary | std::ios::trunc);
        _file_for_key_value.close();

        // Открываем для чтения/записи: .tree без append, .data — с append
        _file_for_tree.open(tree_file,
                            std::ios::in | std::ios::out | std::ios::binary);

        _file_for_key_value.open(data_file,
                                 std::ios::in |
                                 std::ios::out |
                                 std::ios::binary |
                                 std::ios::app); // ← ключевой режим

        // Инициализация
        _count_of_node = 0;
        _position_root = 0;

        btree_disk_node root_node(true);
        root_node.position_in_disk = _count_of_node++;
        _position_root = root_node.position_in_disk;

        disk_write(root_node);

        // Записать заголовок
        _file_for_tree.seekp(0, std::ios::beg);
        _file_for_tree.write(
                reinterpret_cast<const char *>(&_count_of_node),
                sizeof(size_t));
        _file_for_tree.write(
                reinterpret_cast<const char *>(&_position_root),
                sizeof(size_t));
        _file_for_tree.flush();
    } else {
        // Открываем существующие файлы
        _file_for_tree.open(tree_file,
                            std::ios::in | std::ios::out | std::ios::binary);

        _file_for_key_value.open(data_file,
                                 std::ios::in |
                                 std::ios::out |
                                 std::ios::binary |
                                 std::ios::app); // ← и тут

        _file_for_tree.read(
                reinterpret_cast<char *>(&_count_of_node),
                sizeof(size_t));
        _file_for_tree.read(
                reinterpret_cast<char *>(&_position_root),
                sizeof(size_t));

        _current_node = disk_read(_position_root);
    }
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
void B_tree_disk<tkey, tvalue, compare, t>::check_tree(size_t pos, size_t depth) {
    if (pos == static_cast<size_t>(-1)) return;

    btree_disk_node node = disk_read(pos);

    // Проверка количества ключей
    if (pos != _position_root && (node.size < minimum_keys_in_node || node.size > maximum_keys_in_node)) {
        throw std::logic_error("Invalid keys count in node at position " + std::to_string(pos));
    }

    // Проверка порядка ключей
    for (size_t i = 1; i < node.keys.size(); ++i) {
        if (compare_keys(node.keys[i].first, node.keys[i - 1].first)) {
            throw std::logic_error("Keys order violation in node at position " + std::to_string(pos));
        }
    }

    // Рекурсивная проверка дочерних узлов
    if (!node._is_leaf) {
        if (node.pointers.size() != node.size + 1) {
            throw std::logic_error("Pointers count mismatch in node at position " + std::to_string(pos));
        }

        for (size_t i = 0; i < node.pointers.size(); ++i) {
            btree_disk_node child = disk_read(node.pointers[i]);

            // Проверка интервалов ключей
            if (i > 0 && !compare_keys(child.keys.front().first, node.keys[i - 1].first)) {
                throw std::logic_error("Left interval violation in node at position " + std::to_string(pos));
            }

            if (i < node.keys.size() && !compare_keys(node.keys[i].first, child.keys.back().first)) {
                throw std::logic_error("Right interval violation in node at position " + std::to_string(pos));
            }

            check_tree(node.pointers[i], depth + 1);
        }
    }
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator::btree_disk_const_iterator(
        B_tree_disk<tkey, tvalue, compare, t> *tree, const std::stack<std::pair<size_t, size_t> > &path,
        size_t index) : _path(path), _index(index), _tree(tree) {
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
typename B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator
B_tree_disk<tkey, tvalue, compare, t>::begin() {
    if (_position_root == static_cast<size_t>(-1)) {
        return end();
    }

    // Находим самый левый (минимальный) элемент
    std::stack<std::pair<size_t, size_t> > path;
    size_t current_position = _position_root;
    btree_disk_node current_node = disk_read(current_position);

    while (!current_node._is_leaf) {
        path.push({current_position, 0});
        current_position = current_node.pointers[0];
        current_node = disk_read(current_position);
    }

    path.push({current_position, 0});
    return btree_disk_const_iterator(*this, path, 0);
}


// Возвращает итератор, указывающий за последний элемент
template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
typename B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator
B_tree_disk<tkey, tvalue, compare, t>::end() {
    return btree_disk_const_iterator(this, {}, 0);
}

// Префиксный инкремент
template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
typename B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator::self&
B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator::operator++()
{
    // 1) Если итератор уже в end() — ничего не делаем.
    if (_path.empty())
        return *this;

    // 2) Смотрим, куда сейчас «список» итерации указывает сверху стека:
    auto [pos, idx] = _path.top();
    auto node = _tree->disk_read(pos);

    if (!node._is_leaf) {
        // --------------------
        // Случай: мы на внутренней ноде (например, после find_range).
        // Надо просто перейти на следующий ребёнок и опуститься в него в самый левый лист.

        // 2.1) сдвигаем активный указатель «какой ребёнок следующий»
        _path.top().second = idx + 1;

        // 2.2) спускаемся по этому указателю до самого левого листа
        size_t child_pos = node.pointers[idx + 1];
        auto child = _tree->disk_read(child_pos);
        while (!child._is_leaf) {
            _path.push({ child_pos, 0 });
            child_pos = child.pointers[0];
            child = _tree->disk_read(child_pos);
        }
        // пушим сам лист
        _path.push({ child_pos, 0 });
        return *this;
    }

    // --------------------
    // Случай: мы на листе
    // 3) Если в текущем листе ещё остались ключи — просто сдвинуть индекс
    if (idx + 1 < node.keys.size()) {
        _path.top().second = idx + 1;
        return *this;
    }

    // 4) Иначе лист кончился — «восходим» наверх
    _path.pop();
    while (!_path.empty()) {
        auto [parent_pos, parent_idx] = _path.top();
        auto parent = _tree->disk_read(parent_pos);

        // 4.1) Если у этого предка ещё есть ключ, который мы не отдали:
        //      указатель parent_idx прямо указывает на номер ключа.
        if (parent_idx < parent.keys.size()) {
            // оставляем рамку предка как есть — на неё будет указывать
            // operator*() и отдаст parent.keys[parent_idx].
            return *this;
        }
        // end()
        if (parent_idx >= parent.keys.size()) {
            _path.pop();
            return *this;
        }
        // 4.2) Иначе, все ключи в этом узле уже «отданы» — можно проверить,
        //      есть ли у него ещё правое поддерево
        if (parent_idx + 1 < parent.pointers.size()) {
            // 4.2.1) сдвинуть указатель на это правое поддерево
            _path.top().second = parent_idx + 1;

            // 4.2.2) спуститься в его самый левый лист
            size_t child_pos = parent.pointers[parent_idx + 1];
            auto child = _tree->disk_read(child_pos);
            while (!child._is_leaf) {
                _path.push({ child_pos, 0 });
                child_pos = child.pointers[0];
                child = _tree->disk_read(child_pos);
            }
            _path.push({ child_pos, 0 });
            return *this;
        }

        // 4.3) ни ключей, ни правого поддерева — поднимаемся выше
        _path.pop();
    }

    // 5) Стек опустел — это и есть end()
    return *this;
}

// Постфиксный инкремент
template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
typename B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator::self
B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator::operator++(int) {
    self tmp = *this;
    ++(*this);
    return tmp;
}

// Обратные итераторы не поддерживаются
template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
typename B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator::self &
B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator::operator--() {
    throw std::logic_error("Reverse iteration not supported");
}

template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
typename B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator::self
B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator::operator--(int) {
    throw std::logic_error("Reverse iteration not supported");
}

// Сравнение итераторов
template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
bool B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator::operator==(
        self &other) noexcept {
    return _path == other._path;
}

template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
bool B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator::operator!=(
        self &other) noexcept {
    return !(*this == other);
}

// Доступ к элементу
template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
typename B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator::value_type
B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator::operator*() noexcept {
    if (_path.empty()) throw std::out_of_range("Dereferencing end iterator");

    auto [pos, idx] = _path.top();
    auto node = _tree->disk_read(pos);
    return node.keys[idx];
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
std::optional<tvalue> B_tree_disk<tkey, tvalue, compare, t>::at(const tkey &key) {
    auto [path, result] = find_path(key);
    auto [index, found] = result;

    if (!found) {
        return std::nullopt;
    }

    size_t current_position = path.top().first;
    btree_disk_node current_node = disk_read(current_position);

    return current_node.keys[index].second;
}


template<serializable tkey, serializable tvalue, compator<tkey> compare, std::size_t t>
std::pair<typename B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator,
        typename B_tree_disk<tkey, tvalue, compare, t>::btree_disk_const_iterator>
B_tree_disk<tkey, tvalue, compare, t>::find_range(const tkey &lower, const tkey &upper,
                                                  bool include_lower, bool include_upper) {
    // 1) Нижняя граница
    auto [lower_path, lower_res] = find_path(lower);
    auto [li, lower_found]      = lower_res;
    btree_disk_const_iterator it_lo = end();
    if (!lower_path.empty()) {
        it_lo = btree_disk_const_iterator(this, lower_path, li);
        if (lower_found && !include_lower)
            ++it_lo;
    }

    // 2) Верхняя граница — сразу past-the-end
    auto [upper_path, upper_res] = find_path(upper);
    auto [ui, upper_found]      = upper_res;
    btree_disk_const_iterator it_hi = end();
    if (!upper_path.empty()) {
        // hi_index = первый элемент > upper
        size_t hi_index = ui;

        it_hi = btree_disk_const_iterator(this, upper_path, hi_index);
        if (upper_found && !include_upper)
            ++it_hi;
    }

    return {it_lo, it_hi};
}


#endif //B_TREE_DISK_HPP
