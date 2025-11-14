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

    toString() {
        return this.toArray()
            .map(s => `Roll No: ${s.rollNo}, Name: ${s.name}, Grade: ${s.grade}`)
            .join('\n');
    }
}