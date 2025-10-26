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

    deleteByRollNo(rollNo) {
        if (!this.head) return false;

        if (this.head.data.rollNo === rollNo) {
            this.head = this.head.next;
            this._size--;
            return true;
        }

        let prev = null;
        let current = this.head;

        while (current && current.data.rollNo !== rollNo) {
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