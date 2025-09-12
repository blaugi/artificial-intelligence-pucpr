class Node:
    utility:int
    children:list["Node"]

    def get_children(self) -> list["Node"]:
        return self.children
    
    def is_terminal(self) -> bool:
        match len(self.children):
            case 0:
                return True
            case _:
                return False
        
    def get_utility(self) -> int:
        return self.utility
    
    def add_child(self, child:"Node"):
        self.children.append()
    


def main():
    print("Hello from minimax!")


if __name__ == "__main__":
    main()
