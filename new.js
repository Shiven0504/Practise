// ...existing code...
class StudentNode {
    constructor(data) {
        this.data = data;
        this.next = null;
    }
}
// ...existing code...
class LinkedList {
    constructor() {
        this.head = null;
        this._size = 0;
    }

    size() {
        return this._size;
    }

    isEmpty() {
        return this._size === 0;
    }

    insertAt(position, data) {
        if (position < 0) {
            console.error("Invalid position");
            return false;
        }

        const newNode = new StudentNode(data);

        if (position === 0) {
            newNode.next = this.head;
            this.head = newNode;
            this._size++;
            return true;
        }

        let prev = null;
        let current = this.head;
        let index = 0;

        while (current && index < position) {
            prev = current;
            current = current.next;
            index++;
        }

        if (index !== position) {
            console.error("Position out of bounds");
            return false;
        }

        prev.next = newNode;
        newNode.next = current;
        this._size++;
        return true;
    }

    insertEnd(data) {
        return this.insertAt(this._size, data);
    }

    getAt(index) {
        if (index < 0 || index >= this._size) return null;
        let cur = this.head;
        let i = 0;
        while (cur && i < index) {
            cur = cur.next;
            i++;
        }
        return cur ? cur.data : null;
    }

    removeAt(index) {
        if (index < 0 || index >= this._size) return null;
        if (index === 0) {
            const removed = this.head;
            this.head = this.head.next;
            this._size--;
            return removed.data;
        }
        let prev = null;
        let cur = this.head;
        let i = 0;
        while (cur && i < index) {
            prev = cur;
            cur = cur.next;
            i++;
        }
        if (!cur) return null;
        prev.next = cur.next;
        this._size--;
        return cur.data;
    }

    findByRollNo(rollNo) {
        let cur = this.head;
        let index = 0;
        while (cur) {
            if (cur.data && cur.data.rollNo === rollNo) return { node: cur, index };
            cur = cur.next;
            index++;
        }
        return { node: null, index: -1 };
    }

    findIndex(predicate) {
        let cur = this.head;
        let idx = 0;
        while (cur) {
            if (predicate(cur.data, idx)) return idx;
            cur = cur.next;
            idx++;
        }
        return -1;
    }

    deleteByRollNo(rollNo) {
        if (!this.head) return false;

        if (this.head.data && this.head.data.rollNo === rollNo) {
            this.head = this.head.next;
            this._size--;
            return true;
        }

        let prev = null;
        let current = this.head;

        while (current && (!current.data || current.data.rollNo !== rollNo)) {
            prev = current;
            current = current.next;
        }

        if (!current) {
            console.warn("Roll number not found");
            return false;
        }

        prev.next = current.next;
        this._size--;
        return true;
    }

    clear() {
        this.head = null;
        this._size = 0;
    }

    reverse() {
        let prev = null;
        let current = this.head;

        while (current) {
            const next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }

        this.head = prev;
        return this;
    }

    toArray() {
        const out = [];
        let cur = this.head;
        while (cur) {
            out.push(cur.data);
            cur = cur.next;
        }
        return out;
    }

    forEach(fn) {
        let cur = this.head;
        let idx = 0;
        while (cur) {
            fn(cur.data, idx);
            cur = cur.next;
            idx++;
        }
    }

    map(fn) {
        const out = [];
        this.forEach((d, i) => out.push(fn(d, i)));
        return out;
    }

    toString() {
        return this.toArray()
            .map(s => `Roll No: ${s.rollNo}, Name: ${s.name}, Grade: ${s.grade}`)
            .join('\n');
    }

    print() {
        const arr = this.toArray();
        if (arr.length === 0) {
            console.log("List is empty");
            return;
        }
        console.log(`Student List (size=${this.size()}):`);
        arr.forEach(s => console.log(`Roll No: ${s.rollNo}, Name: ${s.name}, Grade: ${s.grade}`));
    }
}
    //